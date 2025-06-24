#!/usr/bin/env python3

import sys, json, time, argparse, keyring
from urllib.parse import urljoin
import requests
from requests.exceptions import RequestException, Timeout
import matplotlib.pyplot as plt
from collections import OrderedDict

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

if not GRAFANA_TOKEN:
    sys.exit("Error: GRAFANA_TOKEN is empty!")

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {GRAFANA_TOKEN}"})


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

    query = f"SELECT \"pedestal\" FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"LEF_name\" =~ /^{lef}$/ AND \"file\" =~ /^{file.replace("/", "\\/")}$/) AND time >= {from_ms}ms AND time <= {now_ms}ms GROUP BY \"channel\" ORDER BY time ASC" 

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

    query = f"SELECT \"raw_sigma\" FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"LEF_name\" =~ /^{lef}$/ AND \"file\" =~ /^{file.replace("/", "\\/")}$/) AND time >= {from_ms}ms AND time <= {now_ms}ms GROUP BY \"channel\" ORDER BY time ASC" 

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

    query = f"SELECT \"sigma\" FROM \"Calibration\" WHERE (\"tag\" =~ /^{tag}$/ AND \"LEF_name\" =~ /^{lef}$/ AND \"file\" =~ /^{file.replace("/", "\\/")}$/) AND time >= {from_ms}ms AND time <= {now_ms}ms GROUP BY \"channel\" ORDER BY time ASC" 

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
    p.add_argument("uid_or_url", nargs="?", default=BASE_DASH_UID,
                   help="Dashboard UID or full https:// URL")
    p.add_argument("--ntags", type=int, default=20,
                   help="How many TAG values to fetch (default: 20)")
    p.add_argument("--time", type=int, default=3,
                   help="How many hours to look back (default: 3 hours)")
    p.add_argument("--plot", action="store_true",
                   help="Plot pedestals, raw sigmas, and sigmas")
    p.add_argument("--print", action="store_true", default=False,
                   help="Print dashboard JSON (default: True)")
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
        
    if args.ntags > 0 and args.time > 0:
        tags = last_tags(args.ntags, args.time)
        print(f"\nLatest {args.ntags} TAG values (last {args.time} h):")
        for t in tags:
            print(f"  • Found tag: {t}")
                    
            fileno = fileno_for_tag(t)
            if fileno[0] == -1 or fileno[0] == -2 or fileno[0] == -3 or fileno[0] == -4:
                print(f"\t  • FileNo not found: {fileno[1]}")
            else:
                    print(f"\t  • FileNo: {fileno[1]}")
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
                            print(f"\t\t\t  • {l}")
                            pedestals = lef_ped(t, l, fileno[1])
                            rsigs = lef_rsig(t, l, fileno[1])
                            sigmas = lef_sig(t, l, fileno[1])
                            
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
                                plt.show()

if __name__ == "__main__":
    main()
