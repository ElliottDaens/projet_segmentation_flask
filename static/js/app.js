/* ═══════════════════════════════════════════════════════════════════
   Segmentation Embarcations — JavaScript principal
   ═══════════════════════════════════════════════════════════════════ */

// ── Thème ───────────────────────────────────────────────────────────

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
}

(function initTheme() {
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
})();

// ── Tutoriel toggle ─────────────────────────────────────────────────

function toggleTuto(el) {
    const box = el.closest('.tuto-box');
    box.classList.toggle('open');
    const key = 'tuto_' + (box.dataset.tutoId || '');
    localStorage.setItem(key, box.classList.contains('open') ? '1' : '0');
}

(function initTutos() {
    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.tuto-box').forEach(box => {
            const key = 'tuto_' + (box.dataset.tutoId || '');
            const saved = localStorage.getItem(key);
            if (saved === '1') box.classList.add('open');
        });
    });
})();

// ── Toast notifications ─────────────────────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
}

// ── Loading overlay ─────────────────────────────────────────────────

function showLoading(text) {
    const overlay = document.getElementById('loading-overlay');
    if (!overlay) return;
    const loadingText = document.getElementById('loading-text');
    if (loadingText) loadingText.textContent = text || 'Traitement en cours…';
    overlay.classList.remove('hidden');
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.add('hidden');
}

// ── Clustering : embeddings ─────────────────────────────────────────

function computeEmbeddings() {
    const method = document.getElementById('embedding-method').value;
    showLoading('Calcul des embeddings (' + method + ')…');

    fetch('/api/compute-embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: method })
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        document.getElementById('embedding-status').innerHTML =
            '<span class="status-ok">✅ Embeddings calculés (' + data.method +
            ') — ' + data.n_images + ' images, ' + data.embedding_dim + 'D</span>';
        showToast('Embeddings calculés avec succès !');
    })
    .catch(err => { hideLoading(); showToast('Erreur : ' + err, 'error'); });
}

// ── Clustering : exécution ──────────────────────────────────────────

function runClustering() {
    const method = document.getElementById('clustering-method').value;
    const payload = { method: method };

    if (method === 'kmeans') {
        payload.n_clusters = parseInt(document.getElementById('n-clusters').value);
    } else {
        payload.eps = parseFloat(document.getElementById('dbscan-eps').value);
        payload.min_samples = parseInt(document.getElementById('dbscan-min-samples').value);
    }

    showLoading('Clustering en cours…');

    fetch('/api/run-clustering', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { showToast(data.error, 'error'); return; }
        renderResults(data.report, data.metrics, data.optimal_k);
        showToast('Clustering terminé !');
    })
    .catch(err => { hideLoading(); showToast('Erreur : ' + err, 'error'); });
}

function runSemiSupervised() {
    const nClusters = parseInt(document.getElementById('n-clusters').value);
    showLoading('Clustering semi-supervisé…');

    fetch('/api/semi-supervised-clustering', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_clusters: nClusters })
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) { showToast(data.error, 'error'); return; }
        renderResults(data.report, data.metrics, null);
        showToast('Clustering semi-supervisé terminé !');
    })
    .catch(err => { hideLoading(); showToast('Erreur : ' + err, 'error'); });
}

// ── Rendu des résultats ─────────────────────────────────────────────

const CLUSTER_COLORS = [
    '#7c3aed', '#2563eb', '#0891b2', '#059669', '#d97706',
    '#dc2626', '#ec4899', '#65a30d', '#6366f1', '#14b8a6'
];

function renderResults(report, metrics, optimalK) {
    document.getElementById('results-section').style.display = '';
    document.getElementById('cluster-grid-section').style.display = '';

    // Scatter plot
    const traces = {};
    report.scatter_data.forEach(pt => {
        const c = pt.cluster;
        if (!traces[c]) {
            traces[c] = { x: [], y: [], text: [], name: 'Cluster ' + c,
                mode: 'markers', type: 'scatter',
                marker: { size: 12, color: CLUSTER_COLORS[c % CLUSTER_COLORS.length], opacity: 0.85 }
            };
        }
        traces[c].x.push(pt.x);
        traces[c].y.push(pt.y);
        traces[c].text.push(pt.filename + (pt.manual_label ? ' [' + pt.manual_label + ']' : ''));
    });

    const plotData = Object.values(traces);
    const plotLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text').trim() },
        xaxis: { title: 'UMAP 1', gridcolor: 'rgba(128,128,128,0.2)' },
        yaxis: { title: 'UMAP 2', gridcolor: 'rgba(128,128,128,0.2)' },
        margin: { t: 20, r: 20, b: 50, l: 50 },
        legend: { x: 1, y: 1, bgcolor: 'rgba(0,0,0,0)' },
        hovermode: 'closest',
    };

    Plotly.newPlot('scatter-plot', plotData, plotLayout, { responsive: true });

    // Métriques
    const metricsPanel = document.getElementById('metrics-panel');
    let metricsHtml = '';
    metricsHtml += metricRow('Clusters', metrics.n_clusters);
    metricsHtml += metricRow('Images', metrics.n_samples);
    if (metrics.silhouette_avg !== undefined)
        metricsHtml += metricRow('Silhouette (moy.)', metrics.silhouette_avg.toFixed(3));
    if (metrics.davies_bouldin !== undefined)
        metricsHtml += metricRow('Davies-Bouldin', metrics.davies_bouldin.toFixed(3));
    if (metrics.n_noise !== undefined)
        metricsHtml += metricRow('Bruit (outliers)', metrics.n_noise);
    if (metrics.cluster_sizes) {
        Object.entries(metrics.cluster_sizes).forEach(([k, v]) => {
            metricsHtml += metricRow('Cluster ' + k, v + ' images');
        });
    }
    metricsPanel.innerHTML = metricsHtml;

    // Méthode du coude
    if (optimalK && optimalK.k_values) {
        const elbowTraces = [
            { x: optimalK.k_values, y: optimalK.inertias, name: 'Inertie', yaxis: 'y' },
            { x: optimalK.k_values, y: optimalK.silhouettes, name: 'Silhouette', yaxis: 'y2' }
        ];
        const elbowLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text').trim(), size: 11 },
            xaxis: { title: 'K', gridcolor: 'rgba(128,128,128,0.2)' },
            yaxis: { title: 'Inertie', side: 'left', gridcolor: 'rgba(128,128,128,0.2)' },
            yaxis2: { title: 'Silhouette', side: 'right', overlaying: 'y', gridcolor: 'rgba(128,128,128,0.2)' },
            margin: { t: 10, r: 60, b: 40, l: 60 },
            legend: { x: 0.5, y: 1.1, orientation: 'h' },
            shapes: [{
                type: 'line', x0: optimalK.best_k, x1: optimalK.best_k,
                y0: 0, y1: 1, yref: 'paper',
                line: { color: '#ef4444', width: 2, dash: 'dash' }
            }],
        };
        Plotly.newPlot('elbow-plot', elbowTraces, elbowLayout, { responsive: true });
    }

    // Grille d'images par cluster
    renderClusterGrid(report);
}

function metricRow(label, value) {
    return '<div class="stat-row"><span class="stat-label">' + label +
           '</span><span class="stat-value">' + value + '</span></div>';
}

function renderClusterGrid(report) {
    const tabBar = document.getElementById('cluster-tabs');
    const grid = document.getElementById('cluster-grid');
    let tabsHtml = '<button class="tab active" onclick="showCluster(-1, this)">Tous</button>';

    Object.keys(report.clusters).forEach(k => {
        const count = report.clusters[k].length;
        tabsHtml += '<button class="tab" onclick="showCluster(' + k + ', this)">Cluster ' + k +
                     ' (' + count + ')</button>';
    });
    tabBar.innerHTML = tabsHtml;

    let gridHtml = '';
    report.scatter_data.forEach(pt => {
        const color = CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length];
        gridHtml += '<div class="grid-item" data-cluster="' + pt.cluster + '" style="border-color:' + color + '">' +
                    '<img src="/api/image/' + pt.filename + '" loading="lazy">' +
                    '<div class="grid-item-info">' +
                    '<span class="item-name">' + pt.filename + '</span>' +
                    '<span class="badge" style="background:' + color + '">Cluster ' + pt.cluster + '</span>' +
                    (pt.manual_label ? '<span class="badge">' + pt.manual_label + '</span>' : '') +
                    '</div></div>';
    });
    grid.innerHTML = gridHtml;
}

function showCluster(clusterId, btn) {
    document.querySelectorAll('#cluster-tabs .tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('#cluster-grid .grid-item').forEach(item => {
        item.style.display = (clusterId === -1 || parseInt(item.dataset.cluster) === clusterId) ? '' : 'none';
    });
}
