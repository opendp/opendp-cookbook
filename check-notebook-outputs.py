"""
Fails if any executed notebook printed to stderr or raised an error.

nbsphinx saves each notebook after executing it, next to the HTML.
Warnings are shown in the rendered page, but they do not fail the build,
so this script checks the saved outputs instead.
"""

import json
import sys
from pathlib import Path


def get_problems(notebook_path):
    """Yields a short description of each stderr or error output in the notebook."""
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            cell_number = f"cell number {cell['execution_count']}"
            if output["output_type"] == "error":
                yield f"{cell_number}: {output['ename']}: {output['evalue']}"
            elif output["output_type"] == "stream" and output["name"] == "stderr":
                yield f"{cell_number} STDERR: " + "".join(output["text"]).strip()


def main():
    build_path = Path(__file__).parent / "build/html"
    notebook_paths = sorted(build_path.rglob("*.ipynb"))
    if not notebook_paths:
        sys.exit(f"No notebooks found under {build_path}: run 'make html' first.")
    failed = False
    for notebook_path in notebook_paths:
        problems = list(get_problems(notebook_path))
        if problems:
            failed = True
            print(notebook_path)
            for problem in problems:
                print(f"\t{problem}")
    if failed:
        sys.exit("Notebook outputs include warnings or errors.")
    print(f"Checked {len(notebook_paths)} notebooks: no warnings or errors.")


if __name__ == "__main__":
    main()
