"""Run the four-state decision-support policy over JSON Lines records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_support_policy import decide, load_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "decision_support_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL cases")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL decisions")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
                decision = decide(case, policy)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid case at input line {line_number}: {exc}") from exc
            destination.write(json.dumps(decision, sort_keys=True) + "\n")
            processed += 1
    print(f"Wrote {processed} audited decisions to {args.output}")


if __name__ == "__main__":
    main()
