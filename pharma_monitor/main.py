#!/usr/bin/env python3
"""
Pharma AI Monitor - CLI
Usage:
  python main.py                  # full run: generate + synthesize + report
  python main.py --dry-run        # skip API calls, use dummy answers
  python main.py --synthesize     # skip generation, re-run synthesis on existing CSV
  python main.py --report         # skip generation+synthesis, rebuild report only
"""

import sys, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    skip_gen = "--synthesize" in args or "--report" in args
    skip_synth = "--report" in args

    if not skip_gen:
        print("=== Step 1: Generating AI answers ===")
        from generate_answers import generate
        generate(dry_run=dry_run)

    if not skip_synth:
        print("\n=== Step 2: Synthesizing key messages ===")
        from synthesize import synthesize
        synthesize()

        print("\n=== Step 2b: Judging answers against PI ===")
        from judge import judge_all
        judge_all()

    print("\n=== Step 3: Building report ===")
    from report import build_report
    out = build_report()

    # copy to Downloads for Windows
    win_path = "pharma_monitor_report.html"
    shutil.copy(out, win_path)
    print(f"\nCopied to {win_path}")
    print("Done.")


if __name__ == "__main__":
    main()
