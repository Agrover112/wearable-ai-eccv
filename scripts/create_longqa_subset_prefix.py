#!/usr/bin/env python3
"""Create a prefix of an existing LongQA subset manifest for smoke tests."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--count", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data = json.loads(args.source.read_text())
data["samples"] = data["samples"][: args.count]
data["n"] = len(data["samples"])
args.output.write_text(json.dumps(data, indent=2) + "\n")
