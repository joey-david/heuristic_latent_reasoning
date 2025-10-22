from __future__ import annotations

from pathlib import Path

from knot.report import generate_reports

REPORT_CONFIG_PATH = Path("configs/report.yaml")


def main() -> None:
    """Loads the reporting config and renders all requested summaries."""
    generate_reports(REPORT_CONFIG_PATH)


if __name__ == "__main__":
    main()
