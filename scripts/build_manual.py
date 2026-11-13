"""Build the RAG knowledge base PDF from the official OPNsense documentation (BSD 2-Clause).

OPNsense is an open-source firewall/router platform; its documentation is licensed under the
2-clause BSD license, which allows redistribution as long as the copyright notice travels with
it (data/manuals/NOTICE-opnsense-docs.txt). Pages are chosen to match what support tickets ask
about: troubleshooting, install/updates, backups, DHCP/DNS, VPN, firewall, gateways, users/2FA.

Each page is printed with headless Chrome and the pages are merged with pypdf. The resulting PDF
is versioned (data/manuals/opnsense-support-manual.pdf) and its SHA-256 pinned in rag/config.py,
so this script only needs to run again to refresh the manual on purpose.

    uv run python scripts/build_manual.py          # macOS, needs Google Chrome
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger("build_manual")

BASE = "https://docs.opnsense.org"
PAGES: list[str] = [
    # troubleshooting (what breaks)
    "troubleshooting/boot",
    "troubleshooting/config_reset",
    "troubleshooting/gateways",
    "troubleshooting/hardening",
    "troubleshooting/network",
    "troubleshooting/openvpn",
    "troubleshooting/password_reset",
    "troubleshooting/performance",
    "troubleshooting/reset_firmware",
    "troubleshooting/webgui",
    # manual (how to do things)
    "manual/install",
    "manual/updates",
    "manual/firmware",
    "manual/backups",
    "manual/interfaces",
    "manual/gateways",
    "manual/multiwan",
    "manual/dhcp",
    "manual/unbound",
    "manual/dynamic_dns",
    "manual/ntpd",
    "manual/wireless",
    "manual/firewall",
    "manual/vpnet",
    "manual/users",
    "manual/two_factor",
]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = Path("data/manuals/opnsense-support-manual.pdf")


def print_page(url: str, dest: Path) -> None:
    subprocess.run(  # noqa: S603 - fixed binary, URL from the static list above
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=8000",
            f"--print-to-pdf={dest}",
            url,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not Path(CHROME).exists():
        logger.error("Google Chrome not found at %s", CHROME)
        return 1
    writer = PdfWriter()
    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(PAGES):
            url = f"{BASE}/{page}.html"
            part = Path(tmp) / f"{i:02d}.pdf"
            print_page(url, part)
            reader = PdfReader(part)
            start = len(writer.pages)
            for p in reader.pages:
                writer.add_page(p)
            writer.add_outline_item(title=page, page_number=start)
            logger.info("%-34s %3d pages", page, len(reader.pages))
        writer.add_metadata({"/Title": "OPNsense support manual (subset of docs.opnsense.org, BSD 2-Clause)"})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("wb") as fh:
            writer.write(fh)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    logger.info("wrote %s: %d pages, sha256=%s", OUT, len(writer.pages), sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
