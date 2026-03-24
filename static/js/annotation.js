/* ═══════════════════════════════════════════════════════════════════
   Outil d'annotation par POLYGONES
   Avec mode plein écran et zoom/pan
   ═══════════════════════════════════════════════════════════════════ */

const canvasImage = document.getElementById('canvas-image');
const canvasOverlay = document.getElementById('canvas-overlay');
const canvasInteract = document.getElementById('canvas-interact');
const ctxImage = canvasImage.getContext('2d');
const ctxOverlay = canvasOverlay.getContext('2d');
const ctxInteract = canvasInteract.getContext('2d');
const wrapper = document.getElementById('canvas-wrapper');

let currentClassId = 1;
let currentColor = '#0077FF';
let currentClassName = 'eau';

let polygons = [];
let currentPoints = [];
let selectedPolyIndex = -1;

let imgOrigW = 0, imgOrigH = 0;
let cW = 0, cH = 0;
let scaleX = 1, scaleY = 1;
let loadedImg = null;

// Zoom & pan
let zoomLevel = 1;
let panX = 0, panY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;
let panStartPanX = 0, panStartPanY = 0;

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;
const CLOSE_RADIUS = 12;
const VERTEX_RADIUS = 5;

// ── Chargement image ────────────────────────────────────────────────

function loadAnnotationImage() {
    const filename = document.getElementById('annotation-image').value;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function () {
        loadedImg = img;
        imgOrigW = img.naturalWidth;
        imgOrigH = img.naturalHeight;

        const container = document.querySelector('.annotation-canvas-area');
        const availW = container.clientWidth - 30;
        const baseScale = availW / imgOrigW;
        cW = Math.round(imgOrigW * baseScale);
        cH = Math.round(imgOrigH * baseScale);
        scaleX = imgOrigW / cW;
        scaleY = imgOrigH / cH;

        [canvasImage, canvasOverlay, canvasInteract].forEach(c => {
            c.width = cW;
            c.height = cH;
        });

        zoomLevel = 1;
        panX = 0;
        panY = 0;
        applyTransform();

        ctxImage.drawImage(img, 0, 0, cW, cH);
        polygons = [];
        currentPoints = [];
        selectedPolyIndex = -1;

        document.getElementById('canvas-hint').textContent =
            `${imgOrigW}×${imgOrigH}px — Molette = zoom, Espace+clic = déplacer`;

        loadExistingAnnotation(filename);
        redraw();
        updatePolygonList();
        updateZoomLabel();
    };
    img.src = '/api/image/' + filename;
}

function loadExistingAnnotation(filename) {
    fetch('/api/load-annotation/' + filename)
        .then(r => { if (!r.ok) throw new Error('none'); return r.json(); })
        .then(data => {
            if (data.polygons && data.polygons.length > 0) {
                polygons = data.polygons.map(p => ({
                    classId: p.class_id,
                    className: p.class_name,
                    color: p.color,
                    points: p.points.map(pt => [pt[0] / scaleX, pt[1] / scaleY]),
                }));
                redraw();
                updatePolygonList();
                showToast(polygons.length + ' polygone(s) chargé(s)');
            }
        })
        .catch(() => {});
}

// ── Zoom & Pan ──────────────────────────────────────────────────────

function applyTransform() {
    wrapper.style.transformOrigin = '0 0';
    wrapper.style.transform = `scale(${zoomLevel}) translate(${panX}px, ${panY}px)`;
    wrapper.style.width = cW + 'px';
    wrapper.style.height = cH + 'px';
}

function updateZoomLabel() {
    const el = document.getElementById('zoom-label');
    if (el) el.textContent = Math.round(zoomLevel * 100) + '%';
}

function setZoom(newZoom, centerX, centerY) {
    newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
    if (centerX !== undefined && centerY !== undefined) {
        // Zoom centré sur le curseur
        panX = centerX / newZoom - centerX / zoomLevel + panX;
        panY = centerY / newZoom - centerY / zoomLevel + panY;
    }
    zoomLevel = newZoom;
    applyTransform();
    updateZoomLabel();
}

function zoomIn() { setZoom(zoomLevel * 1.3); }
function zoomOut() { setZoom(zoomLevel / 1.3); }
function zoomReset() { zoomLevel = 1; panX = 0; panY = 0; applyTransform(); updateZoomLabel(); }

// Molette = zoom centré sur curseur
const scrollContainer = document.querySelector('.annotation-canvas-area');
scrollContainer.addEventListener('wheel', function (e) {
    if (!loadedImg) return;
    e.preventDefault();
    const rect = canvasInteract.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / zoomLevel;
    const my = (e.clientY - rect.top) / zoomLevel;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setZoom(zoomLevel * factor, mx, my);
}, { passive: false });

// Espace + clic = pan
let spaceHeld = false;
document.addEventListener('keydown', function (e) {
    if (e.code === 'Space' && !e.repeat) {
        spaceHeld = true;
        canvasInteract.style.cursor = 'grab';
    }
});
document.addEventListener('keyup', function (e) {
    if (e.code === 'Space') {
        spaceHeld = false;
        isPanning = false;
        canvasInteract.style.cursor = 'crosshair';
    }
});

canvasInteract.addEventListener('mousedown', function (e) {
    if (spaceHeld || e.button === 1) {
        e.preventDefault();
        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        panStartPanX = panX;
        panStartPanY = panY;
        canvasInteract.style.cursor = 'grabbing';
        return;
    }
});

window.addEventListener('mousemove', function (e) {
    if (isPanning) {
        const dx = (e.clientX - panStartX) / zoomLevel;
        const dy = (e.clientY - panStartY) / zoomLevel;
        panX = panStartPanX + dx;
        panY = panStartPanY + dy;
        applyTransform();
    }
});

window.addEventListener('mouseup', function (e) {
    if (isPanning) {
        isPanning = false;
        canvasInteract.style.cursor = spaceHeld ? 'grab' : 'crosshair';
    }
});

// ── Plein écran ─────────────────────────────────────────────────────

function toggleFullscreen() {
    const area = document.querySelector('.annotation-canvas-area');
    if (!document.fullscreenElement) {
        area.requestFullscreen().then(() => {
            area.classList.add('is-fullscreen');
            // Recalculer la taille du canvas pour remplir l'écran
            setTimeout(resizeCanvasToContainer, 100);
        }).catch(() => {});
    } else {
        document.exitFullscreen();
    }
}

document.addEventListener('fullscreenchange', function () {
    const area = document.querySelector('.annotation-canvas-area');
    if (!document.fullscreenElement) {
        area.classList.remove('is-fullscreen');
        setTimeout(resizeCanvasToContainer, 100);
    }
});

function resizeCanvasToContainer() {
    if (!loadedImg) return;
    const container = document.querySelector('.annotation-canvas-area');
    const availW = container.clientWidth - 20;
    const availH = container.clientHeight - 60;
    const scaleW = availW / imgOrigW;
    const scaleH = availH / imgOrigH;
    const baseScale = Math.min(scaleW, scaleH);

    // Sauvegarder les polygones en coordonnées originales
    const savedPolys = polygons.map(p => ({
        ...p,
        points: p.points.map(pt => [pt[0] * scaleX, pt[1] * scaleY]),
    }));
    const savedCurrent = currentPoints.map(pt => [pt[0] * scaleX, pt[1] * scaleY]);

    cW = Math.round(imgOrigW * baseScale);
    cH = Math.round(imgOrigH * baseScale);
    scaleX = imgOrigW / cW;
    scaleY = imgOrigH / cH;

    [canvasImage, canvasOverlay, canvasInteract].forEach(c => {
        c.width = cW;
        c.height = cH;
    });

    zoomLevel = 1;
    panX = 0;
    panY = 0;
    applyTransform();
    updateZoomLabel();

    ctxImage.drawImage(loadedImg, 0, 0, cW, cH);

    // Restaurer les polygones dans les nouvelles coordonnées
    polygons = savedPolys.map(p => ({
        ...p,
        points: p.points.map(pt => [pt[0] / scaleX, pt[1] / scaleY]),
    }));
    currentPoints = savedCurrent.map(pt => [pt[0] / scaleX, pt[1] / scaleY]);

    redraw();
}

// ── Sélection de classe ─────────────────────────────────────────────

function selectClass(id, color, name) {
    currentClassId = id;
    currentColor = color;
    currentClassName = name;
    document.querySelectorAll('.btn-class').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.id) === id);
    });
    const disp = document.getElementById('current-class-display');
    if (disp) { disp.textContent = name.replace(/_/g, ' '); disp.style.color = color; }
}

// ── Position souris → coordonnées canvas ────────────────────────────

function getPos(e) {
    const rect = canvasInteract.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    // rect est déjà transformé par CSS, on convertit en coordonnées canvas internes
    const x = (clientX - rect.left) * (cW / rect.width);
    const y = (clientY - rect.top) * (cH / rect.height);
    return { x, y };
}

// ── Dessin (clic) ───────────────────────────────────────────────────

canvasInteract.addEventListener('click', function (e) {
    if (isPanning || spaceHeld) return;

    const pos = getPos(e);

    if (currentPoints.length >= 3) {
        const first = currentPoints[0];
        const dist = Math.hypot(pos.x - first[0], pos.y - first[1]);
        if (dist < CLOSE_RADIUS / zoomLevel + 5) {
            closePolygon();
            return;
        }
    }

    currentPoints.push([pos.x, pos.y]);
    selectedPolyIndex = -1;
    redrawInteraction();
});

canvasInteract.addEventListener('dblclick', function (e) {
    e.preventDefault();
    if (currentPoints.length >= 3) closePolygon();
});

canvasInteract.addEventListener('contextmenu', function (e) {
    e.preventDefault();
    if (currentPoints.length > 0) {
        currentPoints = [];
        redrawInteraction();
        showToast('Polygone annulé');
    }
});

canvasInteract.addEventListener('mousemove', function (e) {
    if (isPanning) return;
    const pos = getPos(e);
    redrawInteraction(pos);
});

// Touch support
canvasInteract.addEventListener('touchstart', function (e) {
    e.preventDefault();
    const pos = getPos(e);
    if (currentPoints.length >= 3) {
        const first = currentPoints[0];
        if (Math.hypot(pos.x - first[0], pos.y - first[1]) < CLOSE_RADIUS * 1.5) {
            closePolygon();
            return;
        }
    }
    currentPoints.push([pos.x, pos.y]);
    redrawInteraction();
}, { passive: false });

// ── Fermer un polygone ──────────────────────────────────────────────

function closePolygon() {
    polygons.push({
        classId: currentClassId,
        className: currentClassName,
        color: currentColor,
        points: [...currentPoints],
    });
    currentPoints = [];
    redraw();
    updatePolygonList();
}

// ── Rendu ───────────────────────────────────────────────────────────

function redraw() {
    ctxOverlay.clearRect(0, 0, cW, cH);

    polygons.forEach((poly, idx) => {
        if (poly.points.length < 3) return;

        ctxOverlay.beginPath();
        ctxOverlay.moveTo(poly.points[0][0], poly.points[0][1]);
        for (let i = 1; i < poly.points.length; i++) {
            ctxOverlay.lineTo(poly.points[i][0], poly.points[i][1]);
        }
        ctxOverlay.closePath();
        ctxOverlay.fillStyle = hexToRgba(poly.color, 0.35);
        ctxOverlay.fill();
        ctxOverlay.strokeStyle = poly.color;
        ctxOverlay.lineWidth = idx === selectedPolyIndex ? 3 : 2;
        ctxOverlay.stroke();

        poly.points.forEach(pt => {
            ctxOverlay.beginPath();
            ctxOverlay.arc(pt[0], pt[1], VERTEX_RADIUS, 0, Math.PI * 2);
            ctxOverlay.fillStyle = poly.color;
            ctxOverlay.fill();
            ctxOverlay.strokeStyle = '#fff';
            ctxOverlay.lineWidth = 1;
            ctxOverlay.stroke();
        });

        const cx = poly.points.reduce((s, p) => s + p[0], 0) / poly.points.length;
        const cy = poly.points.reduce((s, p) => s + p[1], 0) / poly.points.length;
        ctxOverlay.font = 'bold 13px sans-serif';
        ctxOverlay.textAlign = 'center';
        ctxOverlay.fillStyle = '#fff';
        ctxOverlay.strokeStyle = '#000';
        ctxOverlay.lineWidth = 3;
        const label = poly.className.replace(/_/g, ' ');
        ctxOverlay.strokeText(label, cx, cy);
        ctxOverlay.fillText(label, cx, cy);
    });

    redrawInteraction();
}

function redrawInteraction(mousePos) {
    ctxInteract.clearRect(0, 0, cW, cH);

    if (currentPoints.length === 0) {
        if (mousePos && !spaceHeld) drawCrosshair(mousePos);
        return;
    }

    ctxInteract.beginPath();
    ctxInteract.moveTo(currentPoints[0][0], currentPoints[0][1]);
    for (let i = 1; i < currentPoints.length; i++) {
        ctxInteract.lineTo(currentPoints[i][0], currentPoints[i][1]);
    }
    if (mousePos) ctxInteract.lineTo(mousePos.x, mousePos.y);
    ctxInteract.strokeStyle = currentColor;
    ctxInteract.lineWidth = 2;
    ctxInteract.setLineDash([6, 4]);
    ctxInteract.stroke();
    ctxInteract.setLineDash([]);

    currentPoints.forEach((pt, i) => {
        ctxInteract.beginPath();
        const r = (i === 0 && currentPoints.length >= 3) ? CLOSE_RADIUS : VERTEX_RADIUS;
        ctxInteract.arc(pt[0], pt[1], r, 0, Math.PI * 2);
        ctxInteract.fillStyle = (i === 0) ? '#ffffff' : currentColor;
        ctxInteract.fill();
        ctxInteract.strokeStyle = currentColor;
        ctxInteract.lineWidth = 2;
        ctxInteract.stroke();
    });

    if (currentPoints.length >= 3 && mousePos) {
        const first = currentPoints[0];
        const dist = Math.hypot(mousePos.x - first[0], mousePos.y - first[1]);
        if (dist < CLOSE_RADIUS * 2) {
            ctxInteract.beginPath();
            ctxInteract.arc(first[0], first[1], CLOSE_RADIUS + 4, 0, Math.PI * 2);
            ctxInteract.strokeStyle = '#fff';
            ctxInteract.lineWidth = 2;
            ctxInteract.stroke();
        }
    }

    if (mousePos && !spaceHeld) drawCrosshair(mousePos);
}

function drawCrosshair(pos) {
    ctxInteract.strokeStyle = 'rgba(255,255,255,0.3)';
    ctxInteract.lineWidth = 1;
    ctxInteract.setLineDash([3, 3]);
    ctxInteract.beginPath();
    ctxInteract.moveTo(pos.x, 0); ctxInteract.lineTo(pos.x, cH);
    ctxInteract.moveTo(0, pos.y); ctxInteract.lineTo(cW, pos.y);
    ctxInteract.stroke();
    ctxInteract.setLineDash([]);
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// ── Liste des polygones ─────────────────────────────────────────────

function updatePolygonList() {
    const list = document.getElementById('polygon-list');
    if (!list) return;
    if (polygons.length === 0) {
        list.innerHTML = '<p class="hint">Aucun polygone. Cliquez sur l\'image pour commencer.</p>';
        return;
    }
    list.innerHTML = polygons.map((p, i) =>
        `<div class="poly-item ${i === selectedPolyIndex ? 'selected' : ''}" onclick="selectPolygon(${i})">
            <span class="cls-dot" style="background:${p.color}"></span>
            <span>${p.className.replace(/_/g,' ')}</span>
            <span class="poly-pts">${p.points.length} pts</span>
            <button class="btn-icon" onclick="event.stopPropagation();deletePolygon(${i})" title="Supprimer">✕</button>
        </div>`
    ).join('');
    document.getElementById('polygon-count').textContent = polygons.length;
}

function selectPolygon(idx) {
    selectedPolyIndex = (selectedPolyIndex === idx) ? -1 : idx;
    redraw();
    updatePolygonList();
}

function deletePolygon(idx) {
    polygons.splice(idx, 1);
    selectedPolyIndex = -1;
    redraw();
    updatePolygonList();
}

function deleteSelected() {
    if (selectedPolyIndex >= 0) deletePolygon(selectedPolyIndex);
}

function clearAll() {
    if (polygons.length === 0 && currentPoints.length === 0) return;
    if (!confirm('Supprimer tous les polygones ?')) return;
    polygons = [];
    currentPoints = [];
    selectedPolyIndex = -1;
    redraw();
    updatePolygonList();
}

function undoLastPoint() {
    if (currentPoints.length > 0) {
        currentPoints.pop();
        redrawInteraction();
    } else if (polygons.length > 0) {
        polygons.pop();
        redraw();
        updatePolygonList();
    }
}

// ── Opacité ─────────────────────────────────────────────────────────

function updateOpacity() {
    const val = parseInt(document.getElementById('overlay-opacity').value);
    document.getElementById('opacity-label').textContent = val;
    canvasOverlay.style.opacity = val / 100;
}

// ── Sauvegarde ──────────────────────────────────────────────────────

function saveAnnotation() {
    const filename = document.getElementById('annotation-image').value;
    const polyData = polygons.map(p => ({
        class_id: p.classId,
        class_name: p.className,
        color: p.color,
        points: p.points.map(pt => [
            Math.round(pt[0] * scaleX),
            Math.round(pt[1] * scaleY),
        ]),
    }));

    fetch('/api/save-annotation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: filename,
            polygons: polyData,
            image_width: imgOrigW,
            image_height: imgOrigH,
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) showToast(`Sauvegardé : ${polygons.length} polygone(s) pour ${filename}`);
        else showToast(data.error || 'Erreur', 'error');
    });
}

// ── Entraînement ────────────────────────────────────────────────────

function trainModel() {
    showLoading('Entraînement du U-Net… Cela peut prendre quelques minutes.');
    fetch('/api/train-segmentation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.success) showToast('Modèle entraîné ! Loss : ' + data.final_loss.toFixed(4));
        else showToast(data.error || 'Erreur', 'error');
    })
    .catch(err => { hideLoading(); showToast('Erreur : ' + err, 'error'); });
}

// ── Raccourcis clavier ──────────────────────────────────────────────

document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); return; }
    switch (e.key) {
        case 'Enter': if (currentPoints.length >= 3) closePolygon(); break;
        case 'Escape':
            if (document.fullscreenElement) { document.exitFullscreen(); break; }
            currentPoints = []; redrawInteraction(); break;
        case 'z': if (e.ctrlKey) { e.preventDefault(); undoLastPoint(); } break;
        case 'Delete': case 'Backspace': deleteSelected(); break;
        case 's': if (e.ctrlKey) { e.preventDefault(); saveAnnotation(); } break;
        case 'f': toggleFullscreen(); break;
        case '+': case '=': zoomIn(); break;
        case '-': zoomOut(); break;
        case '0': zoomReset(); break;
    }
});

// ── Init ────────────────────────────────────────────────────────────

if (INITIAL_IMAGE) loadAnnotationImage();
if (SEG_CLASSES.length > 1) selectClass(SEG_CLASSES[1].id, SEG_CLASSES[1].color, SEG_CLASSES[1].name);
