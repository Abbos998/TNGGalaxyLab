"""
01_select_subhalos.py
=====================
Query the TNG API and select subhalos matching the mass filter.
Writes the target list to `subhalo_ids.txt` for the download stage.

Usage:
    python 01_select_subhalos.py [--config batch_config.yaml]

References:
    IllustrisTNG public data release: Nelson et al. 2019 ComAC 6, 2
    API docs: https://www.tng-project.org/data/docs/api/
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml


def get_headers(token: str) -> dict:
    return {"api-key": token}


def get_with_retry(url, headers, params=None, retries=3, backoff=5.0):
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"  [retry {attempt+1}/{retries}] HTTP {r.status_code} — "
                      f"sleeping {backoff}s...")
                time.sleep(backoff)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [retry {attempt+1}/{retries}] {e} — sleeping {backoff}s...")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="batch_config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    token = cfg["api_token"]
    if token == "PASTE_YOUR_TOKEN_HERE":
        print("ERROR: API token not set in batch_config.yaml", file=sys.stderr)
        sys.exit(1)

    sim = cfg["simulation"]
    snap = cfg["snapshot"]
    Mmin = float(cfg["stellar_mass_min"])   # M_sun
    Mmax = float(cfg["stellar_mass_max"])   # M_sun

    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Fetch simulation and snapshot metadata
    # ------------------------------------------------------------------
    #   Simulation endpoint returns cosmological parameters (h, omega_0, ...).
    #   Snapshot endpoint returns redshift + subhalos_url.
    headers = get_headers(token)
    sim_url  = f"http://www.tng-project.org/api/{sim}/"
    snap_url = f"http://www.tng-project.org/api/{sim}/snapshots/{snap}/"
    print(f"[1] Fetching {sim} metadata...")
    sim_meta  = get_with_retry(sim_url, headers)
    snap_meta = get_with_retry(snap_url, headers)
    # TNG API uses "hubble" in some releases; fall back to standard TNG value
    # if the field is missing (e.g. certain access levels).
    h = sim_meta.get("hubble") or sim_meta.get("hubble_h") or 0.6774
    z = snap_meta.get("redshift", 0.0)
    print(f"    h = {h:.4f},  z = {z:.3f}")

    # ------------------------------------------------------------------
    # 2. Convert physical M_* range → TNG code units (10^10 M_sun/h)
    # ------------------------------------------------------------------
    Mmin_code = Mmin * h / 1e10
    Mmax_code = Mmax * h / 1e10
    print(f"[2] Stellar mass filter:")
    print(f"    Physical: [{Mmin:.2e}, {Mmax:.2e}] M_sun")
    print(f"    Code:     [{Mmin_code:.4f}, {Mmax_code:.4f}] (10^10 M_sun/h)")

    # ------------------------------------------------------------------
    # 3. Query subhalos with mass filter + primary flag
    # ------------------------------------------------------------------
    #   mass_stars__gt=X, mass_stars__lt=Y
    #   primary_flag=1  ==> central subhalos only (avoids satellites)
    # We request more than needed and truncate later.
    print(f"[3] Querying subhalo list (up to {cfg['max_subhalos']*3})...")
    params = dict(
        mass_stars__gt=Mmin_code,
        mass_stars__lt=Mmax_code,
        primary_flag=1,
        limit=cfg["max_subhalos"] * 3,
    )
    subhalos_url = f"http://www.tng-project.org/api/{sim}/snapshots/{snap}/subhalos/"
    result = get_with_retry(subhalos_url, headers, params=params)
    all_subhalos = result["results"]
    print(f"    Fetched {len(all_subhalos)} candidates "
          f"(server reports total = {result['count']})")

    # ------------------------------------------------------------------
    # 4. Truncate to max_subhalos and save
    # ------------------------------------------------------------------
    selected = all_subhalos[:cfg["max_subhalos"]]
    ids_file = outdir / "subhalo_ids.txt"
    with open(ids_file, "w") as f:
        f.write(f"# TNG {sim} snapshot {snap}\n")
        f.write(f"# Filter: {Mmin:.2e} <= M_star <= {Mmax:.2e} M_sun (physical)\n")
        f.write(f"# Central subhalos only (primary_flag=1)\n")
        f.write(f"# Selected: {len(selected)} of {result['count']} total matching\n")
        f.write("# columns: subhalo_id  mass_log_msun  url\n")
        for s in selected:
            # TNG list endpoint returns lightweight objects: id, url,
            # optionally mass_log_msun.  Full properties require a per-subhalo GET.
            mass_str = f"{s.get('mass_log_msun', 0.0):.4f}" \
                       if s.get('mass_log_msun') is not None else "unknown"
            f.write(f"{s['id']}\t{mass_str}\t{s['url']}\n")

    # Also save the raw JSON for downstream use
    with open(outdir / "subhalo_metadata.json", "w") as f:
        json.dump({
            "simulation": sim,
            "snapshot": snap,
            "hubble": h,
            "redshift": z,
            "n_selected": len(selected),
            "n_total_matching": result["count"],
            "stellar_mass_range_msun": [Mmin, Mmax],
            "subhalos": selected,
        }, f, indent=2)

    print(f"\n[✓] Selected {len(selected)} subhalos.")
    print(f"    List:     {ids_file}")
    print(f"    Metadata: {outdir / 'subhalo_metadata.json'}")


if __name__ == "__main__":
    main()
