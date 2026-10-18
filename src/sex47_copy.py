"""sex47_copy.py -- verbatim copy of get_record_ids() / load_sex_map() from No.47's
sex_stratified_eval.py (Supplementary_Code_Data). Used only if that module is not
importable from DNR_SRC on this machine."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_record_ids(dataset_key: str, test_base) -> list[str]:
    """Record identifiers in the exact __getitem__/DataLoader(shuffle=False)
    order, so position i here lines up with position i of evaluate_model's
    returned (labels, probs) arrays. Verified against each class's actual
    __getitem__ implementation (2026-09-22), not assumed."""
    if dataset_key == "ptbxl":
        return [str(x) for x in test_base.db.index.tolist()]
    elif dataset_key == "chapman":
        return [Path(p).name for p in test_base.records]
    raise ValueError(f"unknown dataset_key {dataset_key!r}")


def load_sex_map(dataset_key: str, dataset_kwargs: dict, record_ids: list[str]) -> dict[str, str]:
    """{record_id: 'M'|'F'|'U'}. 'U' = present in the split but not cleanly
    Male/Female (blank/Unknown/missing) -- excluded from the male/female
    comparison but counted and reported, never silently dropped."""
    sex_map: dict[str, str] = {}
    if dataset_key == "ptbxl":
        db = pd.read_csv(Path(dataset_kwargs["root"]) / "ptbxl_database.csv", index_col="ecg_id")
        code_map = {0: "M", 1: "F"}  # PTB-XL documentation: 0=male, 1=female
        for rid in record_ids:
            code = db.loc[int(rid), "sex"] if int(rid) in db.index else None
            sex_map[rid] = code_map.get(code, "U")
    elif dataset_key == "chapman":
        root = Path(dataset_kwargs["root"])
        # Build the name -> .hea path lookup once (WFDBRecords is nested in
        # subdirectories) rather than re-walking the tree per record.
        hea_by_name = {p.stem: p for p in root.rglob("*.hea")}
        for rid in record_ids:
            hea_path = hea_by_name.get(rid)
            sex = "U"
            if hea_path is not None:
                for line in hea_path.read_text().splitlines():
                    if line.strip().startswith("#Sex:"):
                        val = line.split(":", 1)[1].strip().lower()
                        sex = "M" if val == "male" else ("F" if val == "female" else "U")
                        break
            sex_map[rid] = sex
    else:
        raise ValueError(f"unknown dataset_key {dataset_key!r}")
    return sex_map
