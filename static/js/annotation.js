/* ═══════════════════════════════════════════════════════════════════
   Outil d'annotation par zones — Canvas de dessin
   L'utilisateur peint les zones (eau, terre, bateau…) sur l'image.
   Le masque est sauvegardé en niveaux de gris (pixel = indice de classe).
   ═══════════════════════════════════════════════════════════════════ */

const canvasImage = document.getElementById('canvas-image');
const canvasOverlay = document.getElementById('canvas-overlay');
const canvasCursor = document.getElementById('canvas-cursor');
const ctxImage = canvasImage.getContext('2d');
const ctxOverlay = canvasOverlay.getContext('2d');
const ctxCursor = canvasCursor.getContext('2d');

let currentClassId = 0;
let currentColor = '#000000';
let brushSize = 20;
let isDrawing = false;
let undoStack = [];
let imgOriginalWidth = 0;
let imgOriginalHeight = 0;
let canvasDisplayWidth = 0;
let canvasDisplayHeight = 0;

// ── Initialisation ──────────────────────────────────────────────────

function loadAnnotationImage() {
    const filename = document.getElementById('annotation-image').value;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function () {
        imgOriginalWidth = img.naturalWidth;
        imgOriginalHeight = img.naturalHeight;

        const maxW = document.getElementById('canvas-wrapper').clientWidth - 20;
        const scale = Math.min(maxW / img.naturalWidth, 700 / img.naturalHeight, 1);
        canvasDisplayWidth = Math.round(img.naturalWidth * scale);
        canvasDisplayHeight = Math.round(img.naturalHeight * scale);

        [canvasImage, canvasOverlay, canvasCursor].forEach(c => {
            c.width = canvasDisplayWidth;
            c.height = canvasDisplayHeight;
        });

        ctxImage.drawImage(img, 0, 0, canvasDisplayWidth, canvasDisplayHeight);
        ctxOverlay.clearRect(0, 0, canvasDisplayWidth, canvasDisplayHeight);
        undoStack = [];

        document.getElementById('canvas-hint').textContent =
            `${imgOriginalWidth}×${imgOriginalHeight}px — Peignez les zones avec le pinceau`;

        // Charger le masque existant s'il y en a un
        loadExistingMask(filename);
    };
    img.src = '/api/image/' + filename;
}

function loadExistingMask(filename) {
    fetch('/api/load-mask/' + filename)
        .then(r => { if (!r.ok) throw new Error('pas de masque'); return r.json(); })
        .then(data => {
            if (data.mask_b64) {
                const maskImg = new Image();
                maskImg.onload = function () {
                    ctxOverlay.drawImage(maskImg, 0, 0, canvasDisplayWidth, canvasDisplayHeight);
                    saveUndoState();
                };
                maskImg.src = 'data:image/png;base64,' + data.mask_b64;
            }
        })
        .catch(() => {});
}

// ── Sélection de classe ─────────────────────────────────────────────

function selectClass(id, color) {
    currentClassId = id;
    currentColor = color;
    document.querySelectorAll('.btn-class').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.id) === id);
    });
}

// ── Pinceau ─────────────────────────────────────────────────────────

function updateBrushSize() {
    brushSize = parseInt(document.getElementById('brush-size').value);
    document.getElementById('brush-size-label').textContent = brushSize;
}

function updateOpacity() {
    const val = parseInt(document.getElementById('overlay-opacity').value);
    document.getElementById('opacity-label').textContent = val;
    canvasOverlay.style.opacity = val / 100;
}

// ── Dessin ──────────────────────────────────────────────────────────

function getPos(e) {
    const rect = canvasOverlay.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };
}

function drawDot(x, y) {
    ctxOverlay.beginPath();
    ctxOverlay.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctxOverlay.fillStyle = currentColor;
    ctxOverlay.fill();
}

function drawLine(x1, y1, x2, y2) {
    ctxOverlay.lineWidth = brushSize;
    ctxOverlay.lineCap = 'round';
    ctxOverlay.strokeStyle = currentColor;
    ctxOverlay.beginPath();
    ctxOverlay.moveTo(x1, y1);
    ctxOverlay.lineTo(x2, y2);
    ctxOverlay.stroke();
}

let lastX = 0, lastY = 0;

canvasOverlay.addEventListener('mousedown', e => {
    isDrawing = true;
    saveUndoState();
    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;
    drawDot(pos.x, pos.y);
});

canvasOverlay.addEventListener('mousemove', e => {
    const pos = getPos(e);
    drawCursorPreview(pos.x, pos.y);
    if (!isDrawing) return;
    drawLine(lastX, lastY, pos.x, pos.y);
    lastX = pos.x;
    lastY = pos.y;
});

canvasOverlay.addEventListener('mouseup', () => { isDrawing = false; });
canvasOverlay.addEventListener('mouseleave', () => { isDrawing = false; clearCursorPreview(); });

// Touch support
canvasOverlay.addEventListener('touchstart', e => {
    e.preventDefault();
    isDrawing = true;
    saveUndoState();
    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;
    drawDot(pos.x, pos.y);
}, { passive: false });

canvasOverlay.addEventListener('touchmove', e => {
    e.preventDefault();
    if (!isDrawing) return;
    const pos = getPos(e);
    drawLine(lastX, lastY, pos.x, pos.y);
    lastX = pos.x;
    lastY = pos.y;
}, { passive: false });

canvasOverlay.addEventListener('touchend', () => { isDrawing = false; });

// ── Curseur ─────────────────────────────────────────────────────────

function drawCursorPreview(x, y) {
    ctxCursor.clearRect(0, 0, canvasCursor.width, canvasCursor.height);
    ctxCursor.beginPath();
    ctxCursor.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctxCursor.strokeStyle = 'white';
    ctxCursor.lineWidth = 2;
    ctxCursor.stroke();
    ctxCursor.beginPath();
    ctxCursor.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctxCursor.strokeStyle = 'black';
    ctxCursor.lineWidth = 1;
    ctxCursor.stroke();
}

function clearCursorPreview() {
    ctxCursor.clearRect(0, 0, canvasCursor.width, canvasCursor.height);
}

// ── Undo ────────────────────────────────────────────────────────────

function saveUndoState() {
    if (undoStack.length > 30) undoStack.shift();
    undoStack.push(ctxOverlay.getImageData(0, 0, canvasDisplayWidth, canvasDisplayHeight));
}

function undoStroke() {
    if (undoStack.length === 0) return;
    const state = undoStack.pop();
    ctxOverlay.putImageData(state, 0, 0);
}

function clearCanvas() {
    saveUndoState();
    ctxOverlay.clearRect(0, 0, canvasDisplayWidth, canvasDisplayHeight);
}

// ── Sauvegarde du masque ────────────────────────────────────────────

function saveMask() {
    const filename = document.getElementById('annotation-image').value;

    // Créer un canvas temporaire à la taille originale de l'image
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = imgOriginalWidth;
    tmpCanvas.height = imgOriginalHeight;
    const tmpCtx = tmpCanvas.getContext('2d');

    // Redessiner l'overlay à la taille originale
    tmpCtx.drawImage(canvasOverlay, 0, 0, imgOriginalWidth, imgOriginalHeight);
    const imgData = tmpCtx.getImageData(0, 0, imgOriginalWidth, imgOriginalHeight);
    const pixels = imgData.data;

    // Créer le masque : chaque pixel RGB → indice de classe le plus proche
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = imgOriginalWidth;
    maskCanvas.height = imgOriginalHeight;
    const maskCtx = maskCanvas.getContext('2d');
    const maskData = maskCtx.createImageData(imgOriginalWidth, imgOriginalHeight);

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2], a = pixels[i + 3];
        let classId = 255; // non annoté

        if (a > 30) {
            classId = findClosestClass(r, g, b);
        }

        // Stocker en niveaux de gris (R=G=B=classId)
        maskData.data[i] = classId;
        maskData.data[i + 1] = classId;
        maskData.data[i + 2] = classId;
        maskData.data[i + 3] = 255;
    }

    maskCtx.putImageData(maskData, 0, 0);

    // Aussi créer un overlay coloré pour visualisation
    const colorCanvas = document.createElement('canvas');
    colorCanvas.width = imgOriginalWidth;
    colorCanvas.height = imgOriginalHeight;
    const colorCtx = colorCanvas.getContext('2d');
    colorCtx.drawImage(canvasOverlay, 0, 0, imgOriginalWidth, imgOriginalHeight);

    const maskB64 = maskCanvas.toDataURL('image/png').split(',')[1];
    const colorB64 = colorCanvas.toDataURL('image/png').split(',')[1];

    fetch('/api/save-mask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: filename,
            mask_b64: maskB64,
            color_b64: colorB64,
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showToast('Masque sauvegardé pour ' + filename);
        } else {
            showToast(data.error || 'Erreur', 'error');
        }
    });
}

function findClosestClass(r, g, b) {
    let bestId = 0;
    let bestDist = Infinity;
    for (const cls of SEG_CLASSES) {
        const cr = parseInt(cls.color.slice(1, 3), 16);
        const cg = parseInt(cls.color.slice(3, 5), 16);
        const cb = parseInt(cls.color.slice(5, 7), 16);
        const dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
        if (dist < bestDist) {
            bestDist = dist;
            bestId = cls.id;
        }
    }
    return bestId;
}

// ── Entraînement ────────────────────────────────────────────────────

function trainModel() {
    showLoading('Entraînement du U-Net en cours… Cela peut prendre quelques minutes.');

    fetch('/api/train-segmentation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            showToast('Modèle entraîné ! Loss finale : ' + data.final_loss.toFixed(4));
        } else {
            showToast(data.error || 'Erreur', 'error');
        }
    })
    .catch(err => { hideLoading(); showToast('Erreur : ' + err, 'error'); });
}

// ── Init ────────────────────────────────────────────────────────────

if (INITIAL_IMAGE) {
    loadAnnotationImage();
}
selectClass(0, SEG_CLASSES[0].color);
