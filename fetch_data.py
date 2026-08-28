"""
Fetch the five benchmark datasets from their upstream sources.

No dataset is redistributed in this repository. Each is downloaded here and
remains subject to its own licence, which is recorded in DATA_LICENCES.md:

  01_freMTPL2_freq.csv        OpenML data id 41214            CC0 1.0
  02_freMTPL2_sev.csv         CASdatasets freMTPL2sev         GPL (>= 2)
  03_wasa_motorcycle.csv      CASdatasets swmotorcycle        GPL (>= 2)
  04_ausprivauto_dejong.csv   CASdatasets ausprivauto0405     GPL (>= 2)
  05_eudirectlapse_demand.csv CASdatasets eudirectlapse       GPL (>= 2)

Usage:
    python fetch_data.py                # fetch anything missing
    python fetch_data.py --force        # re-download everything
    python fetch_data.py --only 03 05   # fetch selected files only

Requires `pyreadr` in addition to requirements.txt, because the CASdatasets
files are distributed as R .rda binaries:

    pip install pyreadr

If pyreadr will not build on your platform, the R fallback is:

    install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/")
    library(CASdatasets); data(swmotorcycle)
    write.csv(swmotorcycle, "03_wasa_motorcycle.csv", row.names = FALSE)
"""

import argparse
import io
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

CAS_BASE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data"

# The CASdatasets and OpenML files are re-serialised from R/ARFF into CSV, so a
# digest would depend on float formatting. They are verified on shape instead.

# key -> (output filename, expected rows, expected columns)
EXPECTED = {
    "01": ("01_freMTPL2_freq.csv", 678013, 12),
    "02": ("02_freMTPL2_sev.csv", 26444, 2),
    "03": ("03_wasa_motorcycle.csv", 64548, 9),
    "04": ("04_ausprivauto_dejong.csv", 67856, 9),
    "05": ("05_eudirectlapse_demand.csv", 23060, 19),
}

CAS_SOURCES = {
    "02": "freMTPL2sev",
    "03": "swmotorcycle",
    "04": "ausprivauto0405",
    "05": "eudirectlapse",
}


def log(msg):
    print(msg, flush=True)


def download(url):
    log("    GET %s" % url)
    with urllib.request.urlopen(url, timeout=300) as resp:
        return resp.read()


def fetch_openml_freq(out_path):
    """freMTPL2freq, OpenML data id 41214, CC0."""
    from sklearn.datasets import fetch_openml

    log("    fetch_openml(data_id=41214)")
    bunch = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    df = bunch.frame
    # OpenML stores IDpol as the index in some sklearn versions.
    if "IDpol" not in df.columns:
        df = df.reset_index().rename(columns={"index": "IDpol"})
    cols = [
        "IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge",
        "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "OpenML 41214 did not provide the expected columns; missing %s. "
            "Got: %s" % (missing, list(df.columns))
        )
    df[cols].to_csv(out_path, index=False)


def fetch_cas(name, out_path):
    """One CASdatasets .rda, converted to CSV. GPL (>= 2) upstream."""
    try:
        import pyreadr
    except ImportError:
        raise RuntimeError(
            "pyreadr is required to read CASdatasets .rda files.\n"
            "    pip install pyreadr\n"
            "See the R fallback in this file's docstring if it will not build."
        )

    blob = download("%s/%s.rda" % (CAS_BASE, name))
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
        tmp.write(blob)
        tmp_path = tmp.name
    try:
        result = pyreadr.read_r(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result:
        raise RuntimeError("no data frame found inside %s.rda" % name)
    df = result[next(iter(result))]
    df.to_csv(out_path, index=False)


def verify(key, path):
    """Check row and column counts without loading the whole file into pandas."""
    expected_name, exp_rows, exp_cols = EXPECTED[key]
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").rstrip("\r")
        n_cols = len(header.split(","))
        n_rows = sum(1 for _ in fh)
    ok = (n_rows == exp_rows) and (n_cols == exp_cols)
    status = "OK " if ok else "MISMATCH"
    log(
        "    %s %s rows=%d (expected %d)  cols=%d (expected %d)"
        % (status, path.name, n_rows, exp_rows, n_cols, exp_cols)
    )
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file already exists")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    choices=sorted(EXPECTED),
                    help="fetch only these keys, e.g. --only 03 05")
    args = ap.parse_args()

    keys = args.only if args.only else sorted(EXPECTED)
    failures = []

    for key in keys:
        filename, _, _ = EXPECTED[key]
        out_path = HERE / filename
        log("[%s] %s" % (key, filename))

        if out_path.exists() and not args.force:
            log("    already present, skipping (use --force to re-download)")
            if not verify(key, out_path):
                failures.append(filename)
            continue

        try:
            if key == "01":
                fetch_openml_freq(out_path)
            else:
                fetch_cas(CAS_SOURCES[key], out_path)
        except Exception as exc:                      # noqa: BLE001
            log("    FAILED: %s" % exc)
            failures.append(filename)
            continue

        if not verify(key, out_path):
            failures.append(filename)

    log("")
    if failures:
        log("Incomplete. Problem files: %s" % ", ".join(sorted(set(failures))))
        log("A row-count mismatch usually means upstream has revised the data.")
        return 1
    log("All requested datasets are present and match the expected shape.")
    log("Run the pipeline in order: 01 to 07, then 08 and 09.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
