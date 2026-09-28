"""
Interactive applicant scorer (CLI).

The real scorer is built in Phase 7: it will prompt for applicant attributes and
return PD, scaled score, and reason codes. For now it points you at the pipeline
runner so the entry point exists and runs cleanly from day one.

Usage
-----
    python app/scorer.py
"""
from __future__ import annotations


def main():
    print(
        "The interactive applicant scorer is built in Phase 7.\n"
        "Until then, run the pipeline to see current progress:\n\n"
        "    python scripts/run_pipeline.py\n"
    )


if __name__ == "__main__":
    main()
