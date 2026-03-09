from __future__ import annotations

import sys
from pathlib import Path

import fitz

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.corpus.oncology.generate import _page_texts, _save_scanned_pdf

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
PAGE_COUNTS = (5, 10, 15)
APPENDIX_PAGE_TEMPLATES = (
    (
        "Scheduling Appendix\n\n"
        "Upcoming visit type: infusion follow-up\n"
        "Requested arrival window: 08:15 to 08:30\n"
        "Transportation note: patient requested rideshare reimbursement form\n"
        "Administrative note: this page does not contain diagnosis, staging, medication, or lab values"
    ),
    (
        "Insurance Authorization Appendix\n\n"
        "Authorization status: approved\n"
        "Authorization code: NVCC-AUTH-2041\n"
        "Payor outreach contact: utilization management desk\n"
        "Administrative note: this page does not contain diagnosis, staging, medication, or lab values"
    ),
    (
        "Infusion Center Preparation Appendix\n\n"
        "Hydration reminder: drink fluids before arrival\n"
        "Parking desk: level P2 kiosk\n"
        "Visitor policy: one support person permitted\n"
        "Administrative note: this page does not contain diagnosis, staging, medication, or lab values"
    ),
    (
        "Records Release Appendix\n\n"
        "Release requested by: tumor board coordinator\n"
        "Delivery channel: secure fax\n"
        "Requested window: within 48 hours\n"
        "Administrative note: this page does not contain diagnosis, staging, medication, or lab values"
    ),
)


def _appendix_page(index: int) -> str:
    template = APPENDIX_PAGE_TEMPLATES[index % len(APPENDIX_PAGE_TEMPLATES)]
    return f"{template}\n\nAppendix page number: {index + 1}"


def _pages_for_count(total_pages: int) -> tuple[str, ...]:
    base_pages = _page_texts()
    if total_pages < len(base_pages):
        raise ValueError(
            f"total_pages must be at least {len(base_pages)} to preserve the oncology fixture"
        )
    filler_count = total_pages - len(base_pages)
    filler_pages = tuple(_appendix_page(index) for index in range(filler_count))
    return base_pages + filler_pages


def _page_count(path: Path) -> int:
    with fitz.open(path) as pdf:
        return len(pdf)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for total_pages in PAGE_COUNTS:
        output_path = GENERATED / f"oncology_scanned_{total_pages:02d}p.pdf"
        _save_scanned_pdf(output_path, _pages_for_count(total_pages))
        actual_pages = _page_count(output_path)
        if actual_pages != total_pages:
            raise RuntimeError(
                f"Expected {total_pages} pages in {output_path.name}, got {actual_pages}"
            )
        print(f"Wrote {output_path} ({actual_pages} pages)")


if __name__ == "__main__":
    main()
