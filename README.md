# opendp-cookbook

The OpenDP Cookbook is a place for the community to show how they are using OpenDP, and for newcomers to pick up ideas.
If you have a recipe to add, please make a PR!

Recipes should be well documented, informatively titled, self-contained `.py` files.
We use the [jupytext light format](https://jupytext.org/formats/scripts/#the-light-format)
so comments will be parsed as markdown on conversion to notebook.

```
$ python3.14 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ ./cook.py
```

The cookbook then be read at `docs/index.html`.
When PRs are merged, Github Actions updates the [cookbook website](https://opendp.github.io/opendp-cookbook/)
