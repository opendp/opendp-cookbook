# opendp-cookbook

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)

[![docs build](https://github.com/opendp/opendp-cookbook/actions/workflows/docs.yml/badge.svg)](https://github.com/opendp/opendp-cookbook/actions/workflows/docs.yml?query=branch%3Amain)

The OpenDP Cookbook is a place for the community to show how they are using OpenDP, and for newcomers to pick up ideas.
If you have a recipe to add, please make a PR!

## Contribute

Thank you for contributing to the OpenDP Cookbook!

We don't plan to do rigorous code reviews here, but we do have a few requirements:

- Contributors should [sign the CLA](https://docs.opendp.org/en/stable/contributing/cla.html).
- Recipes should be self-contained `.py` files. (If you have several related recipes, feel free to make a PR proposing a new directory to contain them, but each script should be able to run independently.)
- We use the [jupytext light format](https://jupytext.org/formats/scripts/#the-light-format)
so comment blocks in the script will be parsed as markdown on conversion to notebook.
- We use precommit to enforce [ruff](https://docs.astral.sh/ruff/) style.
If a contribution looks good in other respects, the maintainers may apply ruff without bothering the author.
- File names should be relatively short; Titles (as a markdown `#` header in the first comment block) should be informative.
- Don't use subprocess, or notebook magic commands, or read or write outside of a temp directory.
- There should be a lot of explanation: Aim for 50/50 code/prose.
- Additional dependencies are not forbidden, but they are discouraged. Additions (via uv) will be evaluated on a case-by-case basis.

Finally, where should the input data come from?
- May be obvious, but don't check-in private data, or even data that could be mistaken for private data!
- If it's possible, a small function that generates a random dataframe is ideal.
- If it fits your analysis, feel free to reference the [example data provided with OpenDP](https://docs.opendp.org/en/stable/api/python/opendp.extras.examples.html).
- The [sklearn datasets](https://sklearn.org/stable/api/sklearn.datasets.html) are a possibility, if necessary.
- Please do not download datasets from the internet in your script: We want to minimize dependencies on outside resources.

## Develop

For local development:

```
$ pip install uv
$ uv sync --dev
$ uv run pre-commit install
$ uv run make html  # Or provide a filename to render just one file.
```

The cookbook can then be seen at `build/html/index.html`.
When PRs are merged, the [cookbook website](https://opendp.github.io/opendp-cookbook/) is automatically updated.

## Tips

**To migrate `.ipynb` files to `.py`**, use nbconvert:

```
$ uv run jupyter nbconvert example.ipynb --to script --output-dir source
```

Use `# +` and `# -` to group adjacent code "paragraphs" into larger cells.

**To add dependencies**, use uv:

```
$ uv add your-new-dependency
```

The base dependencies should suffice to run the notebooks,
while the dev dependencies are needed for the build.

PRs that add dependencies will be evaluated on a case-by-case basis.
