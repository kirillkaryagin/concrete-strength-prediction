"""Download the original UCI archive and create the documented numerical CSV."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from project import FEATURES, ROOT, TARGET

URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"


def download():
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    request = Request(URL, headers={"User-Agent": "ML-Zoomcamp-Concrete-Strength/1.0"})
    with urlopen(request, timeout=60) as response:
        archive = response.read(2_000_001)
    if len(archive) > 2_000_000:
        raise ValueError("Unexpectedly large archive.")
    archive_hash = hashlib.sha256(archive).hexdigest()
    source_path = data_dir / "source.json"
    if source_path.exists():
        expected = json.loads(source_path.read_text())["archive_sha256"]
        if archive_hash != expected:
            raise ValueError("Official archive changed. Preserve and inspect it before updating.")
    with ZipFile(io.BytesIO(archive)) as zipped:
        names = zipped.namelist()
        selected = {}
        for basename in ["Concrete_Data.xls", "Concrete_Readme.txt"]:
            match = [name for name in names if Path(name).name == basename]
            if len(match) != 1:
                raise ValueError(f"Cannot identify {basename} in the archive.")
            if zipped.getinfo(match[0]).file_size > 2_000_000:
                raise ValueError("Unexpectedly large dataset member.")
            selected[basename] = zipped.read(match[0])
    df = pd.read_excel(io.BytesIO(selected["Concrete_Data.xls"]), engine="xlrd")
    if df.shape != (1030, 9):
        raise ValueError(f"Unexpected source shape: {df.shape}")
    original_columns = list(df.columns)
    df.columns = FEATURES + [TARGET]
    csv_bytes = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    if source_path.exists() and json.loads(source_path.read_text())["normalized_csv_sha256"] != csv_hash:
        raise ValueError("CSV normalization changed. Do not replace the checked data silently.")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    for basename, contents in selected.items():
        (raw_dir / basename).write_bytes(contents)
    (data_dir / "concrete.csv").write_bytes(csv_bytes)
    source = {
        "title": "Concrete Compressive Strength", "creator": "I-Cheng Yeh",
        "citation": "Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.",
        "doi": "https://doi.org/10.24432/C5PK67",
        "source_page": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        "download_url": URL, "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": archive_hash,
        "raw_xls_sha256": hashlib.sha256(selected["Concrete_Data.xls"]).hexdigest(),
        "normalized_csv_sha256": csv_hash,
        "original_rows": len(df), "original_columns": original_columns,
        "normalized_columns": list(df.columns),
        "transformation": "Read the first XLS sheet with xlrd, rename columns by original order, write UTF-8 CSV. No rows removed in this file. Training removes exact duplicate rows.",
    }
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps({"rows": len(df), "columns": len(df.columns), "csv_sha256": csv_hash}, indent=2))


if __name__ == "__main__":
    download()
