# ams_grafana_dump

> CLI utility to retrieve DAQ TAG metadata and calibration data from CERN AMS Grafana dashboards.

## Requirements

* Python ≥ 3.8
* Grafana API token with at least *Viewer* access to the dashboard and datasource.
* InfluxDB datasource UID accessible from Grafana.

### Python packages

```
requests
matplotlib
keyring
```

## Usage

```bash
python ams_grafana_dump.py [options]
```


### Options

| Flag | Default | Description |
|------|---------|-------------|
| `uid_or_url` | `SGiOAOw4k04` | Dashboard UID or full `https://` URL |
| `--ntags N` | `20` | Number of TAG rows to retrieve |
| `--time H` | `3`  | Look-back window in hours |
| `--plot` | _off_ | Display Matplotlib plots for pedestal/raw σ/σ |
| `--print` | _off_ | Print the full dashboard JSON payload |
