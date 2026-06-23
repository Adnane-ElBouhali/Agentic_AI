from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any


DEFAULT_PATHS = [
    "s3+cos://ds-hackalton",
    "s3+cos://ds-hackalton/data",
    "s3+cos://ds-hackalton/data/granting_score",
    "s3+cos://ds-hackalton/data/granting-score",
    "s3+cos://ds-hackalton/data/legal-report",
    "s3+cos://ds-hackalton/data/Perimeter_Data_Updated_2025022.csv",
    "s3+cos://ds-hackalton/data/Perimeter_Data_Updated_20250221.csv",
    "s3+cos://ds-hackalton/data/Perimeter_Data_Updated_2025022.parquet",
    "s3+cos://ds-hackalton/data/Perimeter_Data_Updated_20250221.parquet",
    "ds-hackalton",
    "ds-hackalton/data",
    "ds-hackalton/data/granting_score",
    "ds-hackalton/data/granting-score",
    "ds-hackalton/data/legal-report",
    "ds-hackalton/data/Perimeter_Data_Updated_2025022.csv",
    "ds-hackalton/data/Perimeter_Data_Updated_20250221.csv",
    "ds-hackalton/data/Perimeter_Data_Updated_2025022.parquet",
    "ds-hackalton/data/Perimeter_Data_Updated_20250221.parquet",
    "s3+cos://ds-hackathon/data",
    "ds-hackathon/data",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether Sparrow dataset paths can be listed/opened."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional Sparrow paths to test. Defaults to s3+cos and plain ds-hackalton candidates.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=300_000,
        help="Maximum bytes to read from each file-like path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a readable report.",
    )
    args = parser.parse_args()

    paths = args.paths or DEFAULT_PATHS
    result = {
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "env": _env_status(),
        "imports": _import_status(),
        "paths": [],
    }

    create_path = result["imports"].get("_create_path")
    open_file = result["imports"].get("_open_file")

    for path in paths:
        result["paths"].append(
            _check_path(path, create_path, open_file, max_bytes=args.max_bytes)
        )

    _strip_private_imports(result)
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    else:
        _print_report(result)
    return 0


def _env_status() -> dict[str, bool]:
    keys = [
        "SPARROW_FLOW_IO_S3__TENANTS__COS__URL",
        "SPARROW_FLOW_IO_S3__TENANTS__COS__STATIC_CREDENTIALS__ACCESS_KEY",
        "SPARROW_FLOW_IO_S3__TENANTS__COS__STATIC_CREDENTIALS__SECRET_KEY",
        "SPARROW_FLOW_IO_S3__TENANTS__cos__URL",
        "SPARROW_FLOW_IO_S3__TENANTS__cos__STATIC_CREDENTIALS__ACCESS_KEY",
        "SPARROW_FLOW_IO_S3__TENANTS__cos__STATIC_CREDENTIALS__SECRET_KEY",
        "SPARROW_FLOW_IO_S3__TENANTS__default__STATIC_CREDENTIALS__ACCESS_KEY",
        "SPARROW_FLOW_IO_S3__TENANTS__default__STATIC_CREDENTIALS__SECRET_KEY",
        "SPARROW_FLOW_IO_OBJS__ACCESS_KEY",
        "SPARROW_FLOW_IO_OBJS__SECRET_KEY",
        "SPARROW_OBJS_ACCESS_KEY_ID",
        "SPARROW_OBJS_SECRET_ACCESS_KEY",
        "SPARROW_TOKEN",
    ]
    return {key: bool(os.getenv(key)) for key in keys}


def _import_status() -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        from sparrow_flow.io.path import create_path

        status["sparrow_flow.io.path.create_path"] = True
        status["_create_path"] = create_path
    except Exception as exc:
        status["sparrow_flow.io.path.create_path"] = False
        status["create_path_error"] = str(exc)
        status["_create_path"] = None

    try:
        from sparrow_flow.io.functional import open_file

        status["sparrow_flow.io.functional.open_file"] = True
        status["_open_file"] = open_file
    except Exception as exc:
        status["sparrow_flow.io.functional.open_file"] = False
        status["open_file_error"] = str(exc)
        status["_open_file"] = None

    try:
        import pandas as pd

        status["pandas"] = True
        status["_pandas"] = pd
    except Exception as exc:
        status["pandas"] = False
        status["pandas_error"] = str(exc)
        status["_pandas"] = None

    try:
        import pyarrow.parquet as pq

        status["pyarrow.parquet"] = True
        status["_pyarrow_parquet"] = pq
    except Exception as exc:
        status["pyarrow.parquet"] = False
        status["pyarrow_error"] = str(exc)
        status["_pyarrow_parquet"] = None

    return status


def _check_path(path: str, create_path: Any, open_file: Any, max_bytes: int) -> dict[str, Any]:
    item: dict[str, Any] = {"path": path, "create_path": None, "list": None, "open": None}
    if create_path is not None and _is_create_path_uri(path):
        item["create_path"] = _try_create_and_list(path, create_path)
    elif create_path is not None:
        item["create_path"] = {
            "ok": False,
            "skipped": "plain paths are tested with open_file; use s3+cos://... for create_path",
        }
    else:
        item["create_path"] = {"ok": False, "error": "create_path unavailable"}

    if open_file is not None:
        item["open"] = _try_open_and_summarize(path, open_file, max_bytes=max_bytes)
    else:
        item["open"] = {"ok": False, "error": "open_file unavailable"}
    return item


def _is_create_path_uri(path: str) -> bool:
    return str(path).startswith("s3+")


def _try_create_and_list(path: str, create_path: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False}
    try:
        p = create_path(path)
        out["ok"] = True
        out["type"] = type(p).__name__
        out["repr"] = str(p)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    entries = []
    for method_name, pattern in (("iterdir", None), ("glob", "*"), ("rglob", "*")):
        try:
            method = getattr(p, method_name)
        except Exception:
            continue
        try:
            iterator = method(pattern) if pattern is not None else method()
            for idx, child in enumerate(iterator):
                if idx >= 30:
                    break
                entries.append(str(child))
        except Exception as exc:
            out[f"{method_name}_error"] = str(exc)
        if entries:
            out["list_method"] = method_name
            break
    out["entries"] = entries
    return out


def _try_open_and_summarize(path: str, open_file: Any, max_bytes: int) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False}
    try:
        with open_file(path, "rb") as stream:
            data = stream.read(max_bytes)
        out["ok"] = True
        out["bytes_read"] = len(data)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    out.update(_summarize_bytes(path, data))
    return out


def _summarize_bytes(path: str, data: bytes) -> dict[str, Any]:
    lower = path.lower()
    if lower.endswith(".csv"):
        return _summarize_csv(data)
    if lower.endswith(".parquet"):
        return _summarize_parquet(data)
    if lower.endswith((".pkl", ".pickle")):
        return _summarize_pickle(data)
    return _summarize_text(data)


def _summarize_csv(data: bytes) -> dict[str, Any]:
    try:
        import pandas as pd

        df = pd.read_csv(io.BytesIO(data), nrows=10)
        return {
            "kind": "csv",
            "columns": [str(column) for column in df.columns],
            "sample": json.loads(df.head(5).to_json(orient="records", date_format="iso")),
        }
    except Exception as exc:
        summary = _summarize_text(data)
        summary["csv_error"] = str(exc)
        return summary


def _summarize_parquet(data: bytes) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(io.BytesIO(data))
        meta = parquet.metadata
        summary: dict[str, Any] = {
            "kind": "parquet",
            "num_rows": meta.num_rows if meta else None,
            "num_columns": meta.num_columns if meta else None,
            "columns": [str(name) for name in parquet.schema.names[:80]],
        }
        try:
            batch = next(parquet.iter_batches(batch_size=5), None)
            if batch is not None:
                df = batch.to_pandas()
                summary["sample"] = json.loads(
                    df.head(5).to_json(orient="records", date_format="iso")
                )
        except Exception as exc:
            summary["sample_error"] = str(exc)
        return summary
    except Exception as exc:
        return {"kind": "parquet", "error": str(exc)}


def _summarize_pickle(data: bytes) -> dict[str, Any]:
    try:
        obj = pickle.loads(data)
    except Exception as exc:
        return {"kind": "pickle", "error": str(exc)}

    summary: dict[str, Any] = {"kind": "pickle", "type": type(obj).__name__}
    if hasattr(obj, "shape"):
        summary["shape"] = list(obj.shape)
    if hasattr(obj, "columns"):
        summary["columns"] = [str(column) for column in list(obj.columns)[:80]]
    if hasattr(obj, "head"):
        try:
            summary["sample"] = json.loads(
                obj.head(5).to_json(orient="records", date_format="iso")
            )
        except Exception:
            summary["sample"] = str(obj.head(5))[:2000]
    else:
        summary["repr"] = repr(obj)[:2000]
    return summary


def _summarize_text(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    return {"kind": "text", "preview": text[:2000]}


def _strip_private_imports(result: dict[str, Any]) -> None:
    imports = result.get("imports", {})
    for key in list(imports):
        if key.startswith("_"):
            imports.pop(key, None)


def _print_report(result: dict[str, Any]) -> None:
    print("Sparrow data access check")
    print(f"Python: {result['python']}")
    print(f"CWD: {result['cwd']}")
    print("\nEnvironment variables present:")
    for key, present in result["env"].items():
        print(f"  {key}: {'yes' if present else 'no'}")

    print("\nImports:")
    for key, value in result["imports"].items():
        print(f"  {key}: {value}")

    print("\nPath checks:")
    for item in result["paths"]:
        print(f"\n== {item['path']}")
        print(f"  create_path: {item['create_path']}")
        print(f"  open_file: {item['open']}")


if __name__ == "__main__":
    raise SystemExit(main())
