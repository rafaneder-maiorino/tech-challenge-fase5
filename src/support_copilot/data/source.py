"""Single source of truth for the raw dataset: where it lives and what it must hash to."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

HF_REPO = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main"
RAW_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
RAW_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
RAW_DIR = Path("data/raw")

# The 10 queues present in the English slice (see docs/dataset.md). Training fails
# loudly if upstream ever adds, drops or renames one.
QUEUES: tuple[str, ...] = (
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
)
# Queues that route a ticket to the RAG path ("something needs fixing").
TECHNICAL_QUEUES: frozenset[str] = frozenset(
    {"Technical Support", "Product Support", "IT Support", "Service Outages and Maintenance"}
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Return the raw CSV path, downloading it if missing or if its hash does not match."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / RAW_FILE
    if path.exists() and sha256_of(path) == RAW_SHA256:
        logger.info("raw dataset present, sha256 verified: %s", path)
        return path
    if path.exists():
        logger.warning("sha256 mismatch for %s, downloading again", path)
    url = f"{HF_REPO}/{RAW_FILE}"
    logger.info("downloading %s", url)
    with urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed, pinned URL
        path.write_bytes(resp.read())
    actual = sha256_of(path)
    if actual != RAW_SHA256:
        raise RuntimeError(
            f"downloaded file hash {actual} != expected {RAW_SHA256}. "
            "Upstream changed the file; re-run the audit before updating RAW_SHA256."
        )
    logger.info("download ok, sha256 verified")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_raw()


if __name__ == "__main__":
    main()
