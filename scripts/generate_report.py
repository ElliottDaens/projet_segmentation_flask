# -*- coding: utf-8 -*-
"""Génération de la présentation reveal.js à partir du Markdown."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

REVEAL_CDN = "https://cdn.jsdelivr.net/npm/reveal.js@5.1.0"


def md_to_slides(md_path: str, output_path: str):
    """Convertit un fichier Markdown en slides reveal.js.

    Chaque titre ## crée une nouvelle slide.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    import markdown
    html_body = markdown.markdown(content, extensions=["tables", "fenced_code"])

    sections = html_body.split("<h2>")
    slides_html = ""

    if sections[0].strip():
        slides_html += f"<section>{sections[0]}</section>\n"

    for section in sections[1:]:
        slides_html += f"<section><h2>{section}</section>\n"

    page = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Segmentation d'Embarcations — Présentation</title>
    <link rel="stylesheet" href="{REVEAL_CDN}/dist/reveal.css">
    <link rel="stylesheet" href="{REVEAL_CDN}/dist/theme/moon.css">
    <style>
        .reveal {{ font-size: 28px; }}
        .reveal h1 {{ font-size: 2em; color: #7c3aed; }}
        .reveal h2 {{ font-size: 1.5em; color: #a78bfa; }}
        .reveal table {{ font-size: 0.7em; margin: 0 auto; }}
        .reveal th {{ background: #2d2d4a; }}
        .reveal td, .reveal th {{ padding: 0.4em 0.8em; border: 1px solid #444; }}
        .reveal code {{ background: #1a1a2e; padding: 2px 6px; border-radius: 4px; }}
        .reveal pre {{ background: #1a1a2e; padding: 1em; border-radius: 8px; }}
        .reveal ul, .reveal ol {{ text-align: left; }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            {slides_html}
        </div>
    </div>
    <script src="{REVEAL_CDN}/dist/reveal.js"></script>
    <script>Reveal.initialize({{ hash: true }});</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)


def main():
    md_path = os.path.join(config.REPORTS_DIR, "presentation.md")
    html_path = os.path.join(config.REPORTS_DIR, "slides.html")

    if not os.path.isfile(md_path):
        print(f"ERREUR : {md_path} introuvable.")
        print("Vérifiez que le fichier reports/presentation.md existe.")
        return

    print(f"Conversion {md_path} → {html_path}")
    md_to_slides(md_path, html_path)
    print(f"Slides générées : {html_path}")
    print(f"Ouvrez ce fichier dans un navigateur pour voir la présentation.")


if __name__ == "__main__":
    main()
