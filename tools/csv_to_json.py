#!/usr/bin/env python3
"""
CSV → JSON converter with optional dictionary mapping.

Features:
- Accepts a single CSV file or a directory (processes all .csv files).
- Loads dictionary Excel files from a folder (e.g., services/diccionarios) and
  maps codes to labels per sheet name, creating parallel <col>_txt columns.
- Emits NDJSON (one JSON per line) by default, or a JSON array.

Usage examples (PowerShell):
  # Single file to NDJSON, with dictionaries
  python tools/csv_to_json.py --input "services/grupo/datos.csv" --output "outputs/datos.ndjson" --dict-dir "services/diccionarios"

  # Whole folder to NDJSON files in output dir
  python tools/csv_to_json.py --input "services/grupo" --output "outputs/json" --dict-dir "services/diccionarios"

Notes:
- Dictionary Excel sheets must contain columns named "Código" and "Etiqueta".
- Sheet names should match the column names in the CSV that you want to map.
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
import json
from typing import Dict, Any, Optional, Iterable

import pandas as pd


def load_metadata_mappings(metadata_dir: str) -> Dict[str, Dict[Any, Any]]:
    """Load code→label mappings from all .xlsx files in a directory.

    Each Excel file may contain multiple sheets. For every sheet that has the
    columns {"Código", "Etiqueta"}, we build a mapping dict and store it under
    the sheet name.
    """
    mapping_dict: Dict[str, Dict[Any, Any]] = {}
    if not metadata_dir:
        return mapping_dict
    if not os.path.isdir(metadata_dir):
        print(f"[warn] Dictionary directory not found: {metadata_dir}")
        return mapping_dict

    xlsx_files = glob.glob(os.path.join(metadata_dir, "*.xlsx"))
    if not xlsx_files:
        print(f"[warn] No .xlsx dictionary files found in: {metadata_dir}")

    for xlsx in xlsx_files:
        try:
            sheets = pd.read_excel(xlsx, sheet_name=None)
            for sheet_name, df_map in sheets.items():
                if {"Código", "Etiqueta"}.issubset(df_map.columns):
                    df_map = df_map.drop_duplicates(subset=["Código"], keep="first")
                    mapping_dict[sheet_name] = df_map.set_index("Código")["Etiqueta"].to_dict()
        except Exception as e:
            print(f"[warn] Could not read {xlsx}: {e}")
    return mapping_dict


def add_text_mapped_columns(df: pd.DataFrame, mappings: Dict[str, Dict[Any, Any]]) -> pd.DataFrame:
    """Create <col>_txt columns using provided mappings when available.

    If a column has a mapping in `mappings` keyed by the exact column name,
    we create a new '<col>_txt' column with the mapped label; otherwise we
    create '<col>_txt' as string version of the data.
    """
    if not mappings:
        return df

    out = df.copy()
    for col in df.columns:
        if col in mappings:
            txt_col = f"{col}_txt"
            try:
                out[txt_col] = df[col].map(mappings[col])
            except Exception:
                # If mapping fails (e.g., dtype issues), fallback to str
                out[txt_col] = df[col].astype(str)
        # Do not create _txt for every column unconditionally; only mapped ones
    return out


def write_json(df: pd.DataFrame, out_path: str, ndjson: bool = True) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if ndjson:
        # lines=True => one record per line; force_ascii=False preserves accents
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    else:
        # JSON array; be careful with huge files
        df.to_json(out_path, orient="records", force_ascii=False)


def process_csv(
    csv_path: str,
    output_path: str,
    dict_dir: Optional[str],
    add_txt: bool,
    include_cols: Optional[Iterable[str]] = None,
    drop_cols: Optional[Iterable[str]] = None,
    ndjson: bool = True,
    encoding: str = "utf-8-sig",
) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load
    df = pd.read_csv(csv_path, low_memory=False, encoding=encoding)

    # Subset/include/drop
    if include_cols:
        missing = [c for c in include_cols if c not in df.columns]
        if missing:
            print(f"[warn] include-cols missing in CSV: {missing}")
        df = df[[c for c in include_cols if c in df.columns]]
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Mapping
    if add_txt and dict_dir:
        mappings = load_metadata_mappings(dict_dir)
        df = add_text_mapped_columns(df, mappings)

    # Write
    write_json(df, output_path, ndjson=ndjson)
    print(f"[ok] Wrote: {output_path}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert CSV to JSON/NDJSON with optional dictionary mapping.")
    p.add_argument("--input", required=True, help="Path to a CSV file or a directory containing .csv files")
    p.add_argument("--output", required=True, help="Output file path (for single CSV) or output directory (for input dir)")
    p.add_argument("--dict-dir", default=None, help="Directory containing dictionary .xlsx files (e.g., services/diccionarios)")
    p.add_argument("--ndjson", action="store_true", help="Emit NDJSON (one JSON per line). Default: true")
    p.add_argument("--json-array", action="store_true", help="Emit a JSON array instead of NDJSON")
    p.add_argument("--add-txt-cols", action="store_true", help="Add <col>_txt mapped columns when dictionary available")
    p.add_argument("--include-cols", default=None, help="Comma-separated list of columns to include (order respected)")
    p.add_argument("--drop-cols", default=None, help="Comma-separated list of columns to drop")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig)")

    args = p.parse_args(argv)

    ndjson = True
    if args.json_array:
        ndjson = False
    elif args.ndjson:
        ndjson = True

    include_cols = args.include_cols.split(",") if args.include_cols else None
    include_cols = [c.strip() for c in include_cols] if include_cols else None
    drop_cols = args.drop_cols.split(",") if args.drop_cols else None
    drop_cols = [c.strip() for c in drop_cols] if drop_cols else None

    in_path = args.input
    out_path = args.output
    dict_dir = args.dict_dir
    add_txt = bool(args.add_txt_cols)

    if os.path.isdir(in_path):
        # Batch mode: process all CSVs in directory
        if os.path.isfile(out_path):
            print("[error] Output must be a directory when --input is a directory.")
            return 2
        os.makedirs(out_path, exist_ok=True)
        csvs = sorted(glob.glob(os.path.join(in_path, "*.csv")))
        if not csvs:
            print(f"[warn] No CSV files found in directory: {in_path}")
        for csv_file in csvs:
            base = os.path.splitext(os.path.basename(csv_file))[0]
            ext = ".ndjson" if ndjson else ".json"
            out_file = os.path.join(out_path, base + ext)
            process_csv(
                csv_file,
                out_file,
                dict_dir=dict_dir,
                add_txt=add_txt,
                include_cols=include_cols,
                drop_cols=drop_cols,
                ndjson=ndjson,
                encoding=args.encoding,
            )
    else:
        # Single file
        # If output looks like a directory (ends with separator, is a dir, or no recognized extension), mirror filename
        _, out_ext = os.path.splitext(out_path)
        looks_like_dir = (
            out_path.endswith(os.sep)
            or os.path.isdir(out_path)
            or out_ext.lower() not in {".json", ".ndjson", ".jsonl"}
        )
        if looks_like_dir:
            os.makedirs(out_path, exist_ok=True)
            base = os.path.splitext(os.path.basename(in_path))[0]
            ext = ".ndjson" if ndjson else ".json"
            out_file = os.path.join(out_path, base + ext)
        else:
            # Explicit file path with extension
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            out_file = out_path

        process_csv(
            in_path,
            out_file,
            dict_dir=dict_dir,
            add_txt=add_txt,
            include_cols=include_cols,
            drop_cols=drop_cols,
            ndjson=ndjson,
            encoding=args.encoding,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
