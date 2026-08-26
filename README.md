# opendp-cookbook

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)

[![docs build](https://github.com/opendp/opendp-cookbook/actions/workflows/docs.yml/badge.svg)](https://github.com/opendp/opendp-cookbook/actions/workflows/docs.yml?query=branch%3Amain)

The OpenDP Cookbook is a place for the community to show how they are using OpenDP, and for newcomers to pick up ideas.
If you have a recipe to add, please make a PR!

Recipes should be well documented, informatively titled, self-contained `.py` files.
We use the [jupytext light format](https://jupytext.org/formats/scripts/#the-light-format)
so comments will be parsed as markdown on conversion to notebook.

For local development:

```
$ pip install uv
$ uv sync --dev
$ uv run pre-commit install
$ uv run cook.py
```

The cookbook then be seen at `build/html/index.html`.
When PRs are merged, the [cookbook website](https://opendp.github.io/opendp-cookbook/) is automatically updated.

## Tips

**To migrate `.ipynb` files**, use nbconvert:

```
$ jupyter nbconvert example.ipynb --to script --output-dir source
```

You'll need to add `# +` and `# -` in a few places to help with formatting,
and then add the new script to the table of contents.

**To add dependencies**, use uv:

```
$ uv add your-new-dependency
```

The base dependencies should suffice to run the notebooks,
while the dev dependencies are needed for the build.
