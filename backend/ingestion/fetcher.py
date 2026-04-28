from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import httpx
from tqdm import tqdm

from backend.core.config import config


REGULATION_FILES: dict[str, str] = {
    "eu_ai_act": "eu_ai_act.pdf",
    "dpdp": "dpdp_act_2023.pdf",
}

REGULATION_URLS: dict[str, list[str]] = {
    "eu_ai_act": [
        config.eu_ai_act_url,
        "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689&from=EN",
    ],
    "dpdp": [
        config.dpdp_url,
        "https://egazette.gov.in/WriteReadData/2023/248778.pdf",
    ],
}


def fetch_regulation_pdfs(force_redownload: bool = False) -> dict[str, Path]:
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    for key, urls in REGULATION_URLS.items():
        dest = config.raw_data_dir / REGULATION_FILES[key]
        existing_valid = _is_valid_pdf(dest)

        if not force_redownload and existing_valid:
            checksum = _sha256_checksum(dest)
            print(f"✓ {key}: using cached file at {dest} (sha256: {checksum})")
            results[key] = dest
            continue

        print(f"{key}: download required")
        downloaded = False
        for url in urls:
            print(f"{key}: trying {url}")
            try:
                downloaded_path = _download_with_progress(url, dest)
                if not _is_valid_pdf(downloaded_path):
                    print(f"✗ {key}: invalid PDF from {url}")
                    downloaded_path.unlink(missing_ok=True)
                    continue

                downloaded_path.replace(dest)
                checksum = _sha256_checksum(dest)
                print(f"✓ {key}: saved to {dest} (sha256: {checksum})")
                results[key] = dest
                downloaded = True
                break
            except httpx.HTTPStatusError as exc:
                print(
                    f"✗ {key}: HTTP {exc.response.status_code} while fetching {url}"
                )
            except httpx.RequestError as exc:
                print(f"✗ {key}: network error while fetching {url}: {exc}")
            except Exception as exc:
                print(f"✗ {key}: failed to fetch {url}: {exc}")

        if not downloaded:
            if existing_valid:
                print("⚠ Download failed — keeping existing local file")
                results[key] = dest
                continue
            raise RuntimeError(f"{key}: all download sources failed")

    return results


def _download_with_progress(url: str, dest: Path) -> Path:
    temp_path = dest.with_suffix(dest.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with httpx.stream("GET", url, timeout=60.0) as response:
            response.raise_for_status()
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None

            with temp_path.open("wb") as handle, tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as progress:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    handle.write(chunk)
                    progress.update(len(chunk))

        return temp_path
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _is_valid_pdf(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False

    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _sha256_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    for chunk in _read_in_chunks(path):
        digest.update(chunk)
    return digest.hexdigest()


def _read_in_chunks(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk
