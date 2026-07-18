"""Gate for regenerated OOF prediction files.

Walks ``outputs/exp*_predictions/*.json``, tolerates trailing NUL padding from
transfers, understands both prediction schemas (the minimal PredictionLogger
one and exp7/exp15's rich one), and asserts for every file:
  1. no duplicate pid within a fold,
  2. no pid shared across folds (the leakage bug this whole change fixes),
  3. (optional) the unique-pid count matches an expected-cohort map.

Run as ``python -m shared.verify_oof outputs`` (exit non-zero on any violation).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from shared.cohort import assert_oof_no_leakage

# (regex on the "<dir>/<file>" path, expected unique-pid count), most specific
# first - the first match wins. Cohort size is set by the modality combination:
# quad / text+eeg = 107, + one text = 117, + one eeg = 147, clinical(+smiles) = 198.
# Anchored on '_' delimiters so 'exp1' cannot match an 'exp11' filename.
EXPECTED_COUNTS = [
    (r"exp7_predictions/predictions_oof", 107),        # quad headline (7a) + 7b MoE
    (r"exp15_predictions/predictions_oof", 107),       # REVE quad
    (r"exp9_predictions/predictions_oof", 147),        # standalone EEG encoder sweep
    (r"exp11_6b", 147), (r"exp11_(3a|7a)", 107),       # exp11 sub-configs (base leads the name)
    (r"_exp3[ab]_", 107),
    (r"_exp1[ab]_", 117), (r"_exp5b_", 117), (r"_exp6a_", 117),
    (r"_exp2_", 147), (r"_exp5c_", 147), (r"_exp6b_", 147),
    (r"_exp4[ab]_", 198), (r"_exp5a_", 198),
]


def load_json_nul_tolerant(path: Path) -> dict:
    raw = path.read_bytes().replace(b"\x00", b"").strip()
    return json.loads(raw)


def fold_pid_lists(payload: dict) -> list[list[str]]:
    """Per-fold pid lists, for both schemas (both nest pids under 'folds')."""
    return [[str(p) for p in fold["pids"]] for fold in payload.get("folds", [])]


def expected_for(name: str) -> int | None:
    for pattern, n in EXPECTED_COUNTS:
        if re.search(pattern, name):
            return n
    return None


def verify_file(path: Path) -> list[str]:
    """Return a list of problems for one file (empty == passed)."""
    problems: list[str] = []
    try:
        payload = load_json_nul_tolerant(path)
        folds = fold_pid_lists(payload)  # a non-dict/malformed payload raises here
    except Exception as exc:  # noqa: BLE001 - report per-file, never abort the run
        return [f"unreadable/malformed: {exc}"]
    if not folds:
        return ["no folds / pids"]
    try:
        assert_oof_no_leakage(folds)
    except AssertionError as exc:
        problems.append(str(exc))
    unique = {p for fold in folds for p in fold}
    rel = f"{path.parent.name}/{path.name}"
    want = expected_for(rel)
    if want is not None and len(unique) != want:
        problems.append(f"unique pids {len(unique)} != expected {want}")
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("outputs")
    # OOF files only: in-sample predictions have no folds and legitimately
    # repeat pids, so the leakage/cohort checks don't apply to them.
    files = sorted(root.glob("exp*_predictions/predictions_oof*.json"))
    if not files:
        print(f"no prediction files under {root}", file=sys.stderr)
        return 1
    failed = 0
    for path in files:
        problems = verify_file(path)
        rel = f"{path.parent.name}/{path.name}"
        if problems:
            failed += 1
            for p in problems:
                print(f"FAIL {rel}: {p}", file=sys.stderr)
        else:
            print(f"ok   {rel}")
    print(f"\n{len(files) - failed}/{len(files)} files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
