
"""
Used to read demo data
"""

from pathlib import Path

import aiofiles
from anyio import to_thread
from fastapi import HTTPException, status

from src.features.rag.schemas import MaterialResponse, SectionResponse
from src.settings import get_settings


def _root() -> Path:
    """The project root, four parents up from this file."""
    return Path(__file__).resolve().parents[3]


def _numbers(directory: Path, prefix: str) -> list[int]:
    """Every number a file in this directory is named for, in order."""
    numbers: list[int] = []

    for path in directory.glob(f"{prefix}_*.txt"):
        stem = path.stem.removeprefix(f"{prefix}_")
        if stem.isdigit():
            numbers.append(int(stem))

    return sorted(numbers)


async def _read(directory: Path, prefix: str, number: int, noun: str) -> str:
    """The text of one numbered file, or a 404 naming what is actually there."""
    path = directory / f"{prefix}_{number}.txt"

    # Path.is_file touches the disk, and there is no async twin of it.
    if not await to_thread.run_sync(path.is_file):
        available = await to_thread.run_sync(_numbers, directory, prefix)
        known = f"{available[0]}-{available[-1]}" if available else "none"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {noun} numbered {number}. Available: {known}.",
        )

    async with aiofiles.open(path, encoding="utf-8") as handle:
        return (await handle.read()).strip()


async def get_material(number: int) -> MaterialResponse:
    settings = get_settings()
    text = await _read(_root() / settings.materials_dir, "material", number, "material")

    return MaterialResponse(number=number, text=text)


async def get_section(number: int) -> SectionResponse:
    settings = get_settings()
    text = await _read(_root() / settings.sections_dir, "section", number, "section")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return SectionResponse(
        number=number,
        # The heading is the first line, the subject it covers the second.
        title=" - ".join(lines[:2]),
        text=text,
    )
