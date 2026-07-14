"""
02_download_cutouts.py
=======================
Download stellar cutouts for the selected subhalos.  Resumable — skips
files that already exist.  Rate-limited to be nice to TNG servers.

Usage:
    python 02_download_cutouts.py [--config batch_config.yaml]

Cutout endpoint (from TNG API docs):
    GET /api/{sim}/snapshots/{snap}/subhalos/{id}/cutout.hdf5?stars=<fields>

We request only the fields the Stage 4 pipeline needs:
    Coordinates, Masses, Velocities, Potential, GFM_StellarFormationTime
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

# Fields the Stage 4 pipeline actually reads from the HDF5:
STAR_FIELDS = "Coordinates,Masses,Velocities,Potential,GFM_StellarFormationTime"


def download_cutout(subhalo_id, sim, snap, headers, dest_path,
                    max_retries=3, retry_backoff=5.0, request_delay=1.5):
    """Download one cutout HDF5.  Returns True on success, False on 404."""
    url = (f"http://www.tng-project.org/api/{sim}/snapshots/{snap}/"
           f"subhalos/{subhalo_id}/cutout.hdf5")
    params = {"stars": STAR_FIELDS}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params,
                             timeout=120, stream=True)
            if r.status_code == 200:
                # Stream to disk in 64 KiB chunks
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                time.sleep(request_delay)
                return True
            if r.status_code == 404:
                print(f"    [skip] subhalo {subhalo_id}: no cutout available")
                return False
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"    [retry {attempt+1}/{max_retries}] "
                      f"HTTP {r.status_code} — sleep {retry_backoff}s...")
                time.sleep(retry_backoff)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    [retry {attempt+1}/{max_retries}] {e} — sleep {retry_backoff}s...")
            time.sleep(retry_backoff)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="batch_config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    token = cfg["api_token"]
    if token == "PASTE_YOUR_TOKEN_HERE":
        print("ERROR: API token not set", file=sys.stderr)
        sys.exit(1)

    sim = cfg["simulation"]
    snap = cfg["snapshot"]
    outdir = Path(cfg["output_dir"])
    cutouts_dir = Path(cfg["cutouts_dir"])
    cutouts_dir.mkdir(parents=True, exist_ok=True)

    # Load the target ID list
    ids_file = outdir / "subhalo_ids.txt"
    if not ids_file.exists():
        print(f"ERROR: {ids_file} not found. Run 01_select_subhalos.py first.",
              file=sys.stderr)
        sys.exit(1)

    ids = []
    with open(ids_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ids.append(int(line.split()[0]))

    print(f"Downloading {len(ids)} cutouts from {sim} snapshot {snap}...")
    print(f"  Destination: {cutouts_dir}")
    print(f"  Rate limit:  {cfg['request_delay']}s between requests\n")

    headers = {"api-key": token}
    n_ok = n_skip = n_fail = 0

    for i, sid in enumerate(ids, 1):
        dest = cutouts_dir / f"cutout_{sid}.hdf5"
        if dest.exists() and dest.stat().st_size > 1024:
            n_skip += 1
            print(f"  [{i:>4d}/{len(ids)}] subhalo {sid}: already downloaded, "
                  f"skipping ({dest.stat().st_size/1024/1024:.1f} MB)")
            continue

        print(f"  [{i:>4d}/{len(ids)}] subhalo {sid}: downloading...", end=" ")
        ok = download_cutout(
            sid, sim, snap, headers, dest,
            max_retries=int(cfg["max_retries"]),
            retry_backoff=float(cfg["retry_backoff"]),
            request_delay=float(cfg["request_delay"]),
        )
        if ok:
            n_ok += 1
            print(f"OK ({dest.stat().st_size/1024/1024:.1f} MB)")
        else:
            n_fail += 1
            # Remove partial file
            if dest.exists():
                dest.unlink()

    print(f"\n[Done]")
    print(f"  Downloaded:      {n_ok}")
    print(f"  Skipped (cached): {n_skip}")
    print(f"  Failed:          {n_fail}")

    # Summary file for next stage
    manifest = cutouts_dir / "download_manifest.json"
    with open(manifest, "w") as f:
        json.dump({
            "n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail,
            "n_total": len(ids),
        }, f, indent=2)


if __name__ == "__main__":
    main()
