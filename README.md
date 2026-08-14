# opendp-cookbook

The OpenDP Cookbook is a place for the community to show how they are using OpenDP, and for newcomers to pick up ideas.
If you have a recipe to add, please make a PR!

Recipes should be well documented, informatively titled, self-contained `.py` files.
We use the [jupytext light format](https://jupytext.org/formats/scripts/#the-light-format)
so comments will be parsed as markdown on conversion to notebook.

Optionally, if the first comment lines of the notebook contain a JSON object
with `tag_map` and `css_map` keys,
these are used to configure an interface to show and hide different sections.
[Details here.](https://opendp.github.io/dp-wizard-templates/dp_wizard_templates.html#examples-dp_wizard_templatesconverters)

For local development:

```
$ python3.14 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ ./cook.py
```

The cookbook then be read at `docs/index.html`.
When PRs are merged, the [cookbook website](https://opendp.github.io/opendp-cookbook/) is automatically updated.
