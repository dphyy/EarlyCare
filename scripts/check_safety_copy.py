#!/usr/bin/env python3
"""Block diagnosis-style product copy in user-facing EarlyCare surfaces."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "README.md",
    ROOT / "frontend" / "src",
    ROOT / "frontend" / "scripts",
]
SUFFIXES = {".md", ".py", ".ts", ".tsx", ".mjs", ".css"}
BLOCKED_PHRASES = [
    "parkinson's detected",
    "parkinson detected",
    "detected parkinson",
    "concussion detected",
    "detected concussion",
    "disease diagnosis",
    "medical certainty",
    "emergency confirmed",
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS:
        if target.is_file() and target.suffix in SUFFIXES:
            files.append(target)
        elif target.is_dir():
            files.extend(path for path in target.rglob("*") if path.is_file() and path.suffix in SUFFIXES)
    return sorted(files)


def main() -> int:
    violations: list[str] = []
    for path in iter_files():
        relative = path.relative_to(ROOT)
        text = path.read_text(errors="ignore")
        lowered = text.lower()
        for phrase in BLOCKED_PHRASES:
            if phrase in lowered:
                violations.append(f"{relative}: blocked phrase `{phrase}`")

    if violations:
        print("Blocked EarlyCare safety copy found:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("safety copy ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
