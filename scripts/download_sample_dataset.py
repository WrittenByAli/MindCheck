"""Acquire and subsample a Lichess Elite PGN dataset, and write a manifest.

Subtask 1.2.1 deliverable. Run this once a monthly Lichess Elite archive is
available. It can either download an archive from a URL or process a file you
already downloaded, then subsample it to a manageable number of validated
games and emit ``data/raw/manifest.json`` for reproducibility.

The Lichess Elite Database (https://database.nikonoel.fr) keeps standard games
where a 2400+ player faced a 2200+ opponent (bullet excluded), licensed CC0 —
so every game already satisfies MindCheck's `min_player_elo: 2200` filter.

The monthly archives are distributed as ``.zip`` files (each holds one ``.pgn``);
this script reads the ``.pgn`` straight out of the ``.zip`` without extracting it.

Examples
--------
Download a month and subsample it in one step::

    python scripts/download_sample_dataset.py \
        --url https://database.nikonoel.fr/lichess_elite_2025-11.zip \
        --target-games 3000

Process an archive you already downloaded (.zip or a plain .pgn)::

    python scripts/download_sample_dataset.py \
        --input ~/Downloads/lichess_elite_2025-11.zip
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

# Allow running as a plain script (`python scripts/...`) by adding repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.pgn_parser import StreamingPGNParser  # noqa: E402

DEFAULT_OUTPUT = _REPO_ROOT / "data" / "raw" / "lichess_elite_subset.pgn"
DEFAULT_MANIFEST = _REPO_ROOT / "data" / "raw" / "manifest.json"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> Path:
    """Stream-download ``url`` to ``dest`` (no whole-file buffering)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:  # noqa: S310
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"Downloaded {dest.stat().st_size / 1e6:.1f} MB.")
    return dest


def _iter_source_games(parser: StreamingPGNParser, source: Path, max_games: int) -> Iterator:
    """Yield validated games from a .zip (Lichess Elite) or a plain/compressed PGN.

    For a .zip the first ``.pgn`` member is streamed directly from the archive,
    so nothing is extracted to disk.
    """
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            pgn_members = [n for n in archive.namelist() if n.lower().endswith(".pgn")]
            if not pgn_members:
                raise ValueError(f"No .pgn file found inside {source}.")
            print(f"Reading '{pgn_members[0]}' from {source.name} ...")
            with archive.open(pgn_members[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                yield from parser.iter_games(text, max_games=max_games)
    else:
        yield from parser.stream_games(source, max_games=max_games)


def subsample(source: Path, output: Path, target_games: int, min_elo: int, min_plies: int) -> int:
    """Write up to ``target_games`` validated games from ``source`` to ``output``."""
    parser = StreamingPGNParser(min_plies=min_plies, min_elo=min_elo)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as out:
        for game in _iter_source_games(parser, source, target_games):
            out.write(str(game))
            out.write("\n\n")
            written += 1
            if written % 500 == 0:
                print(f"  ... {written} games written")
    print(f"Wrote {written} validated games to {output}.")
    return written


def write_manifest(
    manifest_path: Path,
    output: Path,
    source_url: str,
    total_games: int,
    min_elo: int,
) -> None:
    manifest = {
        "archive_name": "Lichess Elite Database Subset",
        "provider": "Lichess.org / nikonoel (Public Domain)",
        "source_url": source_url,
        "pgn_path": str(output.relative_to(_REPO_ROOT)),
        "min_elo": min_elo,
        "total_games": total_games,
        "license": "CC0 1.0 / Open Data",
        "sha256_checksum": _sha256(output),
        "download_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote manifest to {manifest_path} (sha256={manifest['sha256_checksum'][:12]}...).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Acquire/subsample a Lichess Elite PGN dataset.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="URL of a .pgn / .pgn.zst archive to download.")
    group.add_argument("--input", help="Path to an already-downloaded archive.")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output subset .pgn path.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Output manifest.json path.")
    ap.add_argument("--target-games", type=int, default=3000, help="Max games to keep.")
    ap.add_argument("--min-elo", type=int, default=2200, help="Minimum player Elo.")
    ap.add_argument("--min-plies", type=int, default=20, help="Minimum game length in plies.")
    args = ap.parse_args(argv)

    output = Path(args.output)
    raw_dir = _REPO_ROOT / "data" / "raw"

    if args.url:
        filename = args.url.rsplit("/", 1)[-1] or "lichess_elite_download.pgn"
        source = download(args.url, raw_dir / filename)
        source_url = args.url
    else:
        source = Path(args.input)
        if not source.is_file():
            ap.error(f"--input file not found: {source}")
        source_url = f"local:{source.name}"

    total = subsample(source, output, args.target_games, args.min_elo, args.min_plies)
    if total == 0:
        print("WARNING: no games passed the filters; check the source archive.", file=sys.stderr)
        return 1

    write_manifest(Path(args.manifest), output, source_url, total, args.min_elo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
