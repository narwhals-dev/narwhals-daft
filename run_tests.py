from __future__ import annotations

import subprocess
import sys

TESTS_THAT_NEED_FIX = [
    ,
]

command = [
    "pytest",
    "narwhals/tests",
    "-p",
    "pytest_constructor_override",
    "-p",
    "env",
    "--use-external-constructor",
    "--constructors",
    "daft",
    "-k",
    f"not ({' or '.join(TESTS_THAT_NEED_FIX)})",
]

if len(sys.argv) > 1:
    command.extend(sys.argv[1:])

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print("Exit code:", e.returncode)
    sys.exit(e.returncode)
