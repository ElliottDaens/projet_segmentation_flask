# static/

Fichiers statiques : CSS et JavaScript.

### `css/style.css`

Thème sombre/clair, composants (cards, grilles, formulaires), outil d'annotation
(canvas, polygones), barres de score, tutoriels pliables, responsive.
CSS custom sans framework externe.

### `js/app.js`

Toggle thème, notifications toast, loading overlay, tutoriels,
fonctions clustering (scatter Plotly, grilles par cluster).

### `js/annotation.js`

Outil de dessin par polygones : 3 canvas empilés, zoom molette (GPU-like smooth),
pan Espace+clic, plein écran (F), sauvegarde/chargement API, raccourcis clavier.
API Canvas 2D native sans bibliothèque externe.
