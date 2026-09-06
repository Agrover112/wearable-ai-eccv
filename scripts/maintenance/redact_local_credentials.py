#!/usr/bin/env python3
"""Replace locally recorded cloud credentials with explicit redaction markers."""

from __future__ import annotations

import pathlib
import re
import sys


PATTERNS = (
    re.compile(r"(?im)^(\s*(?:export\s+)?(?:AWS_)?ACCESS_KEY_ID(?:\s*[:=]\s*|\s+))\S+"),
    re.compile(r"(?im)^(\s*(?:export\s+)?(?:AWS_)?SECRET_ACCESS_KEY(?:\s*[:=]\s*|\s+))\S+"),
    re.compile(r"(?im)^(\s*Access key ID(?:\s*[:=]\s*|\s+))\S+"),
    re.compile(r"(?im)^(\s*Secret access key(?:\s*[:=]\s*|\s+))\S+"),
    re.compile(r"(?im)^(\s*(?:export\s+)?AWS_SESSION_TOKEN(?:\s*[:=]\s*|\s+))\S+"),
)


def redact(path: pathlib.Path) -> bool:
    text = path.read_text()
    updated = text
    for pattern in PATTERNS:
        updated = pattern.sub(r"\1[REDACTED]", updated)
    if updated == text:
        return False
    path.write_text(updated)
    return True


def main() -> int:
    for raw_path in sys.argv[1:]:
        path = pathlib.Path(raw_path)
        if not path.is_file():
            continue
        status = "redacted" if redact(path) else "unchanged"
        print(f"{status}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
