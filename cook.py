#!/usr/bin/env python3

from shutil import rmtree
from json import dumps
from pathlib import Path
import mistune
from dp_wizard_templates.converters import (
    convert_from_notebook,
    convert_to_notebook,
)


ROOT = Path(__file__).parent
DOCS = ROOT / 'docs'

def get_md_line(stem):
    github_url = f"https://github.com/opendp/opendp-cookbook/blob/main/recipes/{stem}.py"
    return (
        f"- [{stem}]({stem}.html) "
        f"([as notebook]({stem}.ipynb), "
        f"[as python on github]({github_url}))"
    )

def write_index(stems):
    index_md = (
        (ROOT / 'recipes/index.md').read_text()
        + "\n"
        + "\n".join(get_md_line(stem) for stem in stems)
    )
    body_inner_html = mistune.html(index_md)
    (DOCS / 'index.html').write_text(
        f"""<!DOCTYPE html>
<html>
    <head><title>OpenDP Cookbook</title></head>
    <img src="assets/opendp-logo.png"></img>
    <body>{body_inner_html}</body>
</html>"""
    )

if __name__ == '__main__':
    DOCS.mkdir(exist_ok=True)
    stems = []
    for py_path in (ROOT / 'recipes').glob('*.py'):
        print(f"{py_path.name}...")
        py_src = py_path.read_text()
        stem = py_path.stem

        py_dict = convert_to_notebook(py_src, title=stem, execute=True)
        (DOCS / f"{stem}.ipynb").write_text(dumps(py_dict))

        py_html = convert_from_notebook(py_dict)
        (DOCS / f"{stem}.html").write_text(py_html)

        stems.append(stem)
    write_index(stems)
    rmtree(DOCS / 'assets')
    (ROOT / 'assets').copy_into(DOCS)
    print("done!")

