from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def update_run_tests() -> None:
    # Run pytest to get failing tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "narwhals/tests",
            "-p",
            "pytest_constructor_override",
            "--use-external-constructor",
            "--tb",
            "no",
            "-v",
            "--constructors",
            "daft",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Extract failed test names using regex
    failed_tests = re.findall(
        r"(?:FAILED|ERROR) narwhals/tests/.*\.py::(\w+)\[?", result.stdout
    )

    # Sort and format the test names
    formatted_tests = ",\n    ".join(f'"{t}"' for t in sorted(set(failed_tests)))

    # Read the current run_tests.py
    run_tests_path = Path("run_tests.py")
    content = run_tests_path.read_text(encoding="utf-8")

    # Replace the TESTS_THAT_NEED_FIX content
    new_content = re.sub(
        r"TESTS_THAT_NEED_FIX\s*=\s*\[.*?\]",
        f"TESTS_THAT_NEED_FIX = [\n    {formatted_tests},\n]",
        content,
        flags=re.DOTALL,
    )

    # Write back to run_tests.py
    if new_content != content:
        run_tests_path.write_text(new_content, encoding="utf-8")

        print("Updated run_tests.py with new failing tests")
    else:
        print("Nothing to update")


if __name__ == "__main__":
    update_run_tests()
