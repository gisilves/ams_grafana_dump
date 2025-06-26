#!/usr/bin/env python3

import sys, json, time, argparse, keyring
from urllib.parse import urljoin
import requests
from requests.exceptions import RequestException, Timeout
import matplotlib.pyplot as plt
import signal
from collections import OrderedDict
import datetime
import os

# ── Constants ───────────────────────────────────────────────────────
GRAFANA_TOKEN = keyring.get_password("grafana", "default")
if not GRAFANA_TOKEN:
    sys.exit("GRAFANA_TOKEN not set - aborting")
BASE_DASH_UID = "SGiOAOw4k04"          # default dashboard UID
DS_UID        = "aemcsd2xsvim8e"       # InfluxDB datasource UID
BASE_HOST     = "https://ams-ami.web.cern.ch"
TIMEOUT       = 60                     # seconds
MAX_REDIRECTS = 20
TAG_RUN       = 2                      # WHERE "run" = 2  (CAL)
TAG_LOOKBACK  = 3                      # hours
TAG_COL_IDX   = 1                      # hard-coded "tag" column
# ────────────────────────────────────────────────────────────────────────────


# ── Mapping for QLs to LEFs ─────────────────────────────────────────────────
ql_mapping = {
    # -------- LEFT QLs --------
    "QL-L1": [
        "12F36", "12F38", "12F43",
        "12F35", "12F19", "10F10",
        "10F07", "08F06", "08F02",
    ],
    "QL-L2": [
        "12F41", "12F14", "12F24",
        "12F18", "12F06", "10F03",
        "10F14", "08F10", "08F13",
    ],
    "QL-L3": [
        "12F30", "12F26", "12F09",
        "12F42", "12F05", "10F13",
        "10F05", "08F08", "08F07",
    ],
    "QL-L4": [
        "12F33", "12F12", "12F31",
        "12F15", "12F17", "10F12",
        "10F04", "08F09", "08F16",
    ],

    # -------- RIGHT QLs --------
    "QL-R1": [
        "12F37", "12F40", "12F39",
        "12F21", "12F20", "10F16",
        "10F11", "08F12", "08F11",
    ],
    "QL-R2": [
        "12F04", "12F10", "12F34",
        "12F02", "12F27", "10F08",
        "10F06", "08F15", "08F04",
    ],
    "QL-R3": [
        "12F16", "12F03", "12F29",
        "12F46", "12F45", "10F18",
        "10F17", "08F05", "08F18",
    ],
    "QL-R4": [
        "12F01", "12F28", "12F08",
        "12F11", "12F25", "10F15",
        "10F02", "08F17", "08F14",
    ],
}


# ────────────────────────────────────────────────────────────────────────────


if not GRAFANA_TOKEN:
    sys.exit("Error: GRAFANA_TOKEN is empty!")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {GRAFANA_TOKEN}"})



def signal_handler(sig, frame):
    # Ask the user if they want to exit
    ask = input("Are you sure you want to exit? [y/N] ")
    if ask.lower() == "y":
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# ─────────────────────────── HTTP helpers ───────────────────────────────────
def fetch_with_trusted_redirects(url: str) -> requests.Response:
    """Follow redirects manually so the Bearer header stays attached."""
    for _ in range(MAX_REDIRECTS):
        try:
            resp = session.get(url, allow_redirects=False, timeout=TIMEOUT)
        except Timeout:
            sys.exit(f"[timeout] GET {url} exceeded {TIMEOUT}s")
        except RequestException as exc:
            sys.exit(f"[http] {exc}")

        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, resp.headers["Location"])
            continue
        return resp
    sys.exit(f"[redirect] more than {MAX_REDIRECTS} redirects – aborting.")


def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except ValueError:
        snippet = resp.text.strip().replace("\n", " ")[:300]
        sys.exit(f"[json] Expected JSON, got: {snippet}")


# ────────────────────── TAG lookup helper ────────────────────────────────────
def last_tags(n: int, lookback: int):
    """
    Return up to the last *n* TAG strings whose timestamp (column 0) sits inside
    the TAG_LOOKBACK window.
    """
    query = (
        f'SELECT "tag" FROM "DAQ_runs" '
        f'WHERE "run" = {TAG_RUN} ORDER BY time DESC LIMIT {n}'
    )
    
    now_ms  = int(time.time() * 1000)

    if lookback:
        from_ms = int(time.time() * 1000) - lookback * 60 * 60 * 1000
    else:
        from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1
        
    payload = {
        "queries": [{
            "refId":      "A",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }

    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["A"]["frames"]
    except RequestException as exc:
        sys.exit(f"[http] {exc}")
    except KeyError:
        sys.exit("[parse] Unexpected Grafana response structure (no frames)")
        
    if not frames:
        sys.exit("[parse] Empty result set")
        
    frame = frames[0]

    # Arrow frame → values per column; table → rows. We assume:
    #   col 0 = time (ms), col 1 = tag
    if "values" in frame["data"]:                     # Arrow layout
        times_ms = frame["data"]["values"][0]
        tags     = frame["data"]["values"][TAG_COL_IDX]
    else:                                             # Table layout
        times_ms = [row[0]            for row in frame["rows"]]
        tags     = [row[TAG_COL_IDX]  for row in frame["rows"]]

    # Filter by look-back window
    filtered = [
        tag for t_ms, tag in zip(times_ms, tags)
        if t_ms >= from_ms
    ]

    return filtered[:n]   # respect n even if more rows match

# ────────────────── TAG info lookup helper ──────────────────────────────────
def tag_info(tag, file):
    
    if not tag:
        return {}

    now_ms  = int(time.time() * 1000)
    from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1
    
    query = f"SELECT LAST(\"nevents\") FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"file\" =~ /^{file.replace('/', '\\/')}$/) AND time >= {TAG_LOOKBACK * 60 * 60 * 1000}ms AND time <= {int(time.time() * 1000)}ms GROUP BY \"test\"" 
    
    payload = {
        "queries": [{
            "refId":      "G",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }

    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["G"]["frames"]        
    except RequestException as exc:
        sys.exit(f"[http] {exc}")
    except KeyError:
        sys.exit("[parse] Unexpected Grafana response structure (no frames)")

    if not frames:
        sys.exit("[parse] Empty result set")

    
    testname = frames[0]['schema']['fields'][1]['labels']['test']
    
    timestamp = frames[0]["data"]["values"][0][0]
    # Convert to human-readable string
    timestamp = datetime.datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    nevents = frames[0]["data"]["values"][1][0]
    
    
    return [0, testname, timestamp, nevents]
    
# ────────────────── FileNo lookup helper ────────────────────────────────────
def fileno_for_tag(tag):
    """
    Return a dict {tag → FileNo} for every tag in *tags_needed*.
    """
    if not tag:
        return {}

    query = (
        f'SELECT "file" ' 
        f'FROM "DAQ_runs" '
        f'WHERE "tag" =~ /^({tag})$/ '
        f'ORDER BY time DESC LIMIT 1'
    )

    now_ms  = int(time.time() * 1000)
    from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1

    payload = {
        "queries": [{
            "refId":      "B",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }


    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["B"]["frames"]
    except RequestException as exc:
        print(f"\t[http] {exc}")
        return [-1, exc]
    except KeyError:
        print("\t[parse] Unexpected Grafana response structure (no frames)")
        return [-2, "KeyError"]
    
    if not frames:
        print("\t[parse] Empty result set")
        return [-3, "Empty result set"]
    
    frame = frames[0]
    fileno = frame["data"]["values"][1][0]
    
    if not fileno:
        return [-4, "FileNo not found"]
    
    return [0, fileno]

# ────────────────── LEF lookup helper ────────────────────────────────────
def lef_for_file(tag, fileno):
    """
    Return a dict {FileNo → LEF}
    """
    if not fileno:
        return {}

    query = (
        f'SELECT "lef" '
        f'FROM "Calibration" '
        f'WHERE "tag" =~ /^({tag})$/ '
    )

    now_ms  = int(time.time() * 1000)
    from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1

    payload = {
        "queries": [{
            "refId":      "C",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }
    
    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["C"]["frames"]
        
    except RequestException as exc:
        print(f"\t[http] {exc}")
        return [-1, exc]
    except KeyError:
        print("\t[parse] Unexpected Grafana response structure (no frames)")
        return [-2, "KeyError"]

    if not frames:
        print("\t[parse] Empty result set")
        return [-3, "Empty result set"]
    
    frame = frames[0]
    lef = frame["data"]["values"][1]
    
    if not lef:
        return [-4, "LEF not found"]    
    
    return [0, lef]

# ────────────────── PEDESTAL lookup helper ────────────────────────────────────
def lef_ped(tag, lef, file):
    """
    Return ordered list of pedestals for a given tag, LEF, and file.
    """
    if not lef:
        return {}
    
    now_ms  = int(time.time() * 1000)
    from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1

    query = f"SELECT \"pedestal\" FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"LEF_name\" =~ /^{lef}$/ AND \"file\" =~ /^{file.replace("/", "\\/")}$/) GROUP BY \"channel\" ORDER BY time ASC" 
    
    payload = {
        "queries": [{
            "refId":      "E",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }

    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["E"]["frames"]
    except RequestException as exc:
        sys.exit(f"[http] {exc}")
    except KeyError:
        sys.exit("[parse] Unexpected Grafana response structure (no frames)")

    if not frames:
        sys.exit("[parse] Empty result set")

    ordered_dict = OrderedDict(sorted(
        (
            (int(fld["labels"]["channel"]), float(vals[0]))       # (channel, pedestal)
            for frame in frames
            for fld, vals in zip(frame["schema"]["fields"], frame["data"]["values"])
            if fld.get("labels", {}).get("channel") and
               fld["config"].get("displayNameFromDS", "").startswith("Calibration.pedestal")
        ),
        key=lambda kv: kv[0]                                      # sort once, by channel
    ))
    
    return list(ordered_dict.values())

# ────────────────── RAW SIGMA lookup helper ────────────────────────────────────
def lef_rsig(tag, lef, file):
    """
    Return ordered list of raw sigmas for a given tag, LEF, and file.
    """
    if not lef:
        return {}
     
    now_ms  = int(time.time() * 1000)
    from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1

    query = f"SELECT \"raw_sigma\" FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"LEF_name\" =~ /^{lef}$/ AND \"file\" =~ /^{file.replace("/", "\\/")}$/) GROUP BY \"channel\" ORDER BY time ASC" 

    payload = {
        "queries": [{
            "refId":      "F",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }

    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["F"]["frames"]
    except RequestException as exc:
        sys.exit(f"[http] {exc}")
    except KeyError:
        sys.exit("[parse] Unexpected Grafana response structure (no frames)")

    if not frames:
        sys.exit("[parse] Empty result set")

    ordered_dict = OrderedDict(sorted(
        (
            (int(fld["labels"]["channel"]), float(vals[0]))       # (channel, raw_sigma)
            for frame in frames
            for fld, vals in zip(frame["schema"]["fields"], frame["data"]["values"])
            if fld.get("labels", {}).get("channel") and
               fld["config"].get("displayNameFromDS", "").startswith("Calibration.raw_sigma")
        ),
        key=lambda kv: kv[0]                                      # sort once, by channel
    ))
    
    return list(ordered_dict.values())

# ────────────────── SIGMA lookup helper ────────────────────────────────────
def lef_sig(tag, lef, file):
    """
    Return ordered list of sigmas for a given tag, LEF, and file.
    """
    
    if not lef:
        return {}
    
    now_ms  = int(time.time() * 1000)
    from_ms = now_ms - TAG_LOOKBACK * 60 * 60 * 1000 - 1

    query = f"SELECT \"sigma\" FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"LEF_name\" =~ /^{lef}$/ AND \"file\" =~ /^{file.replace("/", "\\/")}$/) GROUP BY \"channel\" ORDER BY time ASC" 

    payload = {
        "queries": [{
            "refId":      "F",
            "datasource": {"uid": DS_UID},
            "rawQuery":   True,
            "query":      query,
            "format":     "table"
        }],
        "from": str(from_ms),
        "to":   str(now_ms)
    }

    url = f"{BASE_HOST}/api/ds/query"
    try:
        resp = session.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        frames = resp.json()["results"]["F"]["frames"]
    except RequestException as exc:
        sys.exit(f"[http] {exc}")
    except KeyError:
        sys.exit("[parse] Unexpected Grafana response structure (no frames)")

    if not frames:
        sys.exit("[parse] Empty result set")

    ordered_dict = OrderedDict(sorted(
        (
            (int(fld["labels"]["channel"]), float(vals[0]))       # (channel, sigma)
            for frame in frames
            for fld, vals in zip(frame["schema"]["fields"], frame["data"]["values"])
            if fld.get("labels", {}).get("channel") and
               fld["config"].get("displayNameFromDS", "").startswith("Calibration.sigma")
        ),
        key=lambda kv: kv[0]                                      # sort once, by channel
    ))
    
    return list(ordered_dict.values())

# ───────────────────────────── CLI ──────────────────────────────────────────
def cli_args():
    p = argparse.ArgumentParser(
        description="Fetch dashboard JSON and most-recent TAG values.")
    p.add_argument("--uid_or_url", nargs="?", default=BASE_DASH_UID,
                   help="Dashboard UID or full https:// URL")
    p.add_argument("--ntags", type=int, default=20,
                   help="How many TAG values to fetch (default: 20)")
    p.add_argument("--time", type=int, default=3,
                   help="How many hours to look back (default: 3 hours)")
    p.add_argument("--plot", action="store_true",
                   help="Plot pedestals, raw sigmas, and sigmas")
    p.add_argument("--print", action="store_true", default=False,
                   help="Print dashboard JSON (default: True)")
    p.add_argument("--newest", type=str, default=256,
                   help="Only fetch TAGs older than the one provided")
    p.add_argument("--oldest", type=str, default=-1,
                   help="Only fetch TAGs newer than the one provided")
    return p.parse_args()


# ───────────────────────────── main ─────────────────────────────────────────
def main():
    args = cli_args()

    dash_url = (args.uid_or_url if args.uid_or_url.startswith("http")
                else f"{BASE_HOST}/api/dashboards/uid/{args.uid_or_url}")

    resp = fetch_with_trusted_redirects(dash_url)
    try:
        resp.raise_for_status()
    except RequestException as exc:
        sys.exit(f"[http] {exc}")

    dash_json = safe_json(resp)
    
    if args.print:
        print(json.dumps(dash_json, indent=2))

    if args.newest != -1:
        print(f"\nDumping TAGs older than {args.newest}")
        
    if args.oldest != -1:
        print(f"\nDumping TAGs newer than {args.oldest}")

    if args.ntags > 0 and args.time > 0:
        tags = last_tags(args.ntags, args.time)
        print(f"\nSearching for latest {args.ntags} TAG values (in the last {args.time} h):")
        for t in tags:
            if args.newest != 256 or args.oldest != -1:
                if int(t[-2:],16) >= int(args.newest[-2:],16):
                    print(f"  • Skipping {t} because it's newer than {args.newest}")
                    continue
                elif int(t[-2:],16) <= int(args.oldest[-2:],16):
                    print(f"  • Skipping {t} because it's older than {args.oldest}")
                    continue

            print(f"  • Found tag: {t}")
            
            # Create output directory
            outdir = f"output/TAG_{t}"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            fileno = fileno_for_tag(t)
            
            if fileno[0] == -1 or fileno[0] == -2 or fileno[0] == -3 or fileno[0] == -4:
                print(f"\t  • FileNo not found: {fileno[1]}")
            else:

                info = tag_info(t, fileno[1])
            
                if info[0] == -1 or info[0] == -2 or info[0] == -3 or info[0] == -4:
                    print("\t  • TAG info not found")
                else:
                    #Print TAG info
                    print(f"\t  • FileNo: {fileno[1]}")
                    print(f"\t  • Test: {info[1]}")
                    print(f"\t  • Timestamp: {info[2]}")
                    print(f"\t  • Events: {info[3]}")
                    print(f"\t  • FileNo: {fileno[1]}")
                    
                    
                    
                    # Test name will be in the format QL_<firstQL>_..._<lastQL>_Test: let's extract all the QLs in order
                    QL_list = info[1].split("_")
                    QL_list.remove("Test")
                    QL_list.remove("QL")
                    
                    print(f"\t  • QLs: {QL_list}")

                    # Create folder for Test
                    testfolder = f"{outdir}/{info[1]}"
                    if not os.path.exists(testfolder):
                        os.makedirs(testfolder)
        
                lef = lef_for_file(t, fileno[1])
                if lef[0] == -1 or lef[0] == -2 or lef[0] == -3 or lef[0] == -4:
                    print(f"\t\t  • LEF not found: {lef[1]}")
                else:
                    # Find unique LEFs
                    lefs = set(lef[1])
                    # Order by LEF name
                    lefs = sorted(lefs, key=lambda x: x.lower())
                    # Remove entries with -B suffix
                    lefs = [l for l in lefs if not l.endswith("-B")]
                    print("\t\tFound " + str(len(lefs)) + " LEFs:")
                    for l in lefs:
                        
                        # Create folder for each LEF 
                        leffolder = f"{testfolder}/{l}"
                        if not os.path.exists(leffolder):
                            os.makedirs(leffolder)
                        
                        # Get the LEF serial number from LEF name and QL mapping
                        lefname = l.split("-")
                        lefname.remove("A")
                        lefname.remove("LEF")
                        lef_serial = ql_mapping["QL-" + QL_list[int(lefname[1])]][int(lefname[2])]
                        
                        print(f"\t\t\t  • {l} - LEF for QL-" + QL_list[int(lefname[1])] + " - " + lef_serial)                        
                        
                        pedestals = lef_ped(t, l, fileno[1])
                        rsigs = lef_rsig(t, l, fileno[1])
                        sigmas = lef_sig(t, l, fileno[1])
                        
                        # Save pedestals to file
                        with open(f"{leffolder}/" + lef_serial +"P.csv", "w") as f:
                            f.write("\"Time\",\"channel\",\"pedestal\"\n")
                            for p in pedestals:
                                f.write(f"{info[2]},{pedestals.index(p)},{p}\n")
                        print(f"\t\t\t\t  • Saved Pedestals to {leffolder}/" + lef_serial +"P.csv")
                                
                        # Save raw sigmas to file
                        with open(f"{leffolder}/" + lef_serial +"R.csv", "w") as f:
                            f.write("\"Time\",\"channel\",\"raw_sigma\"\n")
                            for r in rsigs:
                                f.write(f"{info[2]},{rsigs.index(r)},{r}\n")
                        print(f"\t\t\t\t  • Saved Raw Sigmas to {leffolder}/" + lef_serial +"R.csv")
                                
                        # Save sigmas to file
                        with open(f"{leffolder}/" + lef_serial +"S.csv", "w") as f:
                            f.write("\"Time\",\"channel\",\"sigma\"\n")
                            for s in sigmas:
                                f.write(f"{info[2]},{sigmas.index(s)},{s}\n")
                        print(f"\t\t\t\t  • Saved Sigmas to {leffolder}/" + lef_serial +"S.csv")
            
                        if args.plot:
                            # Plot the list of pedestals, raw sigmas, and sigmas on three subplots
                            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
                            axs[0].plot(pedestals, label="Pedestals")
                            axs[0].set_title("Pedestals", loc="left")
                            axs[1].plot(rsigs, label="Raw Sigmas")
                            axs[1].set_title("Raw Sigmas", loc="left")
                            axs[2].plot(sigmas, label="Sigmas")
                            axs[2].set_title("Sigmas", loc="left")
                            axs[0].legend()
                            axs[1].legend()
                            axs[2].legend()
                            
                            # Save plot to file
                            plt.savefig(f"{leffolder}/calibration.png")
                            print(f"\t\t\t\t  • Saved plot to {leffolder}/calibration.png")
                            plt.close()


if __name__ == "__main__":
    main()
