"""
Fails if any executed notebook printed to stderr or raised an error.

nbsphinx saves each notebook after executing it, next to the HTML.
Warnings are shown in the rendered page, but they do not fail the build,
so this script checks the saved outputs instead.

Usage: python check_notebook_outputs.py [build/html]
"""

import json
import sys
from pathlib import Path


def problems(notebook_path):
    """Yields a short description of each stderr or error output in the notebook."""
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                yield f"{output['ename']}: {output['evalue']}"
            elif output["output_type"] == "stream" and output["name"] == "stderr":
                yield "".join(output["text"]).strip()


def main(build_dir):
    notebook_paths = sorted(Path(build_dir).rglob("*.ipynb"))
    if not notebook_paths:
        sys.exit(f"No notebooks found under {build_dir}: run 'make html' first.")
    failed = False
    for notebook_path in notebook_paths:
        for problem in problems(notebook_path):
            failed = True
            print(f"{notebook_path}:\n    {problem}\n")
    if failed:
        sys.exit("Notebook outputs include warnings or errors.")
    print(f"Checked {len(notebook_paths)} notebooks: no warnings or errors.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/html")
