from pathlib import Path
import mistune

if __name__ == '__main__':
    root = Path(__file__).parent
    index_md = (root / 'recipes/index.md').read_text()
    body_inner_html = mistune.html(index_md)
    (root / 'docs').mkdir(exist_ok=True)
    (root / 'docs/index.html').write_text(
        f"""<!DOCTYPE html>
<html>
    <head><title>OpenDP Cookbook</title></head>
    <body>{body_inner_html}</body>
</html>"""
    )
