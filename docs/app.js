/* app.js ─────────────────────────────────────────────────────────────────────
 * Material Substitution Model — interactive dashboard
 * Requires: Plotly.js (CDN, loaded in index.html)
 * ─────────────────────────────────────────────────────────────────────────── */
"use strict";

// ── Colour palette ─────────────────────────────────────────────────────────
const PALETTE = [
  "#3b82f6","#10b981","#f59e0b","#ef4444","#a78bfa",
  "#06b6d4","#f97316","#84cc16","#ec4899","#6366f1",
];

// ── Axis labels per graph type ─────────────────────────────────────────────
const AXES = {
  copper_price:   { x: "Copper Price ($/tonne)",   y: "Cu Product Market Share" },
  aluminum_price: { x: "Aluminum Price ($/tonne)", y: "Cu Product Market Share" },
  ratio:          { x: "Cu / Al Price Ratio",       y: "Cu Product Market Share" },
};

// ── Shared mutable state ───────────────────────────────────────────────────
const S = {
  data:         null,          // loaded product JSON
  graphType:    "copper_price",
  selRegions:   new Set(),
  showObserved: true,
  showFits:     false,
};

// ── Tiny DOM helpers ───────────────────────────────────────────────────────
const $   = id  => document.getElementById(id);
const fmt = v   => (v == null ? "—" : typeof v === "number" ? v.toFixed(4) : String(v));
const cap = str => str.replace(/_/g," ").replace(/\b\w/g, c => c.toUpperCase());

function colorFor(region) {
  const idx = (S.data?.regions ?? []).indexOf(region);
  return PALETTE[idx % PALETTE.length];
}

// ── Fetch ──────────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} — ${url}`);
  return res.json();
}

// ── Boot ───────────────────────────────────────────────────────────────────
async function init() {
  try {
    const manifest = await fetchJSON("data/manifest.json");
    if (!manifest.products?.length) { showState("empty"); return; }
    buildProductButtons(manifest.products);
    await loadProduct(manifest.products[0]);
  } catch (e) {
    console.error(e);
    showState("empty");
    $("state-empty").innerHTML =
      `<p class="muted">Could not load manifest.<br>
       Run <code>python export_data.py</code>, commit <code>docs/data/</code>,
       then refresh.</p>`;
  }
}

// ── Product buttons ────────────────────────────────────────────────────────
function buildProductButtons(products) {
  const el = $("product-selector");
  el.innerHTML = "";
  products.forEach(p => {
    const btn = document.createElement("button");
    btn.className   = "prod-btn";
    btn.textContent = cap(p);
    btn.dataset.p   = p;
    btn.addEventListener("click", () => loadProduct(p));
    el.appendChild(btn);
  });
}

// ── Load product data ──────────────────────────────────────────────────────
async function loadProduct(product) {
  document.querySelectorAll(".prod-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.p === product));

  showState("loading");
  try {
    S.data       = await fetchJSON(`data/${product}.json`);
    S.selRegions = new Set(S.data.regions);
    buildRegionList(S.data.regions);
    $("header-subtitle").textContent =
      `${cap(product)}  ·  ${S.data.regions.length} regions`;
    renderAll();
    showState("content");
  } catch (e) {
    console.error(e);
    $("state-empty").innerHTML =
      `<p class="muted">Failed to load <code>${product}.json</code>: ${e.message}</p>`;
    showState("empty");
  }
}

// ── Region checkboxes ──────────────────────────────────────────────────────
function buildRegionList(regions) {
  const el = $("region-list");
  el.innerHTML = "";
  regions.forEach(r => {
    const label  = document.createElement("label");
    label.className = "reg-item";
    label.innerHTML = `
      <input type="checkbox" value="${r}" checked />
      <span class="reg-dot" style="background:${colorFor(r)}"></span>
      ${r}`;
    label.querySelector("input").addEventListener("change", e => {
      e.target.checked ? S.selRegions.add(r) : S.selRegions.delete(r);
      renderChart();
    });
    el.appendChild(label);
  });
}

// ── Render everything ──────────────────────────────────────────────────────
function renderAll() {
  renderChart();
  renderFitTable();
  renderSanity();
}

// ── Chart ──────────────────────────────────────────────────────────────────
function renderChart() {
  if (!S.data) return;

  const gType   = S.graphType;
  const gData   = S.data.graphs[gType];
  const labels  = AXES[gType];
  const regions = S.data.regions.filter(r => S.selRegions.has(r));
  const traces  = [];

  // ── Model curves ──────────────────────────────────────────────────────
  regions.forEach(r => {
    const d = gData[r];
    const c = colorFor(r);

    traces.push({
      x: d.x, y: d.y,
      type: "scatter", mode: "lines", name: r,
      line: { color: c, width: 2.2 },
      hovertemplate:
        `<b>${r}</b><br>${labels.x}: %{x:.1f}<br>${labels.y}: %{y:.3f}<extra></extra>`,
    });

    // Observed point (price graphs only)
    if (S.showObserved && d.observed_x != null) {
      traces.push({
        x: [d.observed_x], y: [d.observed_y],
        type: "scatter", mode: "markers",
        name: `${r} (obs)`, showlegend: false,
        marker: { color: c, size: 9, symbol: "circle",
                  line: { color: "#fff", width: 1.5 } },
        hovertemplate:
          `<b>${r} — observed</b><br>${labels.x}: %{x:.1f}<br>${labels.y}: %{y:.3f}<extra></extra>`,
      });
    }
  });

  // ── Fit curves (ratio tab only) ────────────────────────────────────────
  if (S.showFits && gType === "ratio") {
    const { s_min, s_max } = S.data.fit_bounds;

    regions.forEach(r => {
      const fit = S.data.fit_results.find(f => f.region === r);
      if (!fit) return;
      const xs = S.data.graphs.ratio[r].x;
      const c  = colorFor(r);

      // Linear (polynomial degree 1)
      if (fit.poly_a != null) {
        traces.push({
          x: xs, y: xs.map(x => fit.poly_a + fit.poly_b * x),
          type: "scatter", mode: "lines",
          name: `${r} linear`, showlegend: false,
          line: { color: c, width: 1.2, dash: "dot" },
          hovertemplate: `<b>${r} linear</b><br>%{x:.2f}: %{y:.3f}<extra></extra>`,
        });
      }

      // Exponential ("power" in Python)
      if (fit.power_alpha != null) {
        traces.push({
          x: xs, y: xs.map(x => fit.power_alpha * Math.exp(fit.power_beta * x)),
          type: "scatter", mode: "lines",
          name: `${r} exp`, showlegend: false,
          line: { color: c, width: 1.2, dash: "dash" },
          hovertemplate: `<b>${r} exp</b><br>%{x:.2f}: %{y:.3f}<extra></extra>`,
        });
      }

      // Logit
      if (fit.logit_alpha != null) {
        traces.push({
          x: xs,
          y: xs.map(x =>
            s_min + (s_max - s_min) /
            (1 + Math.exp(fit.logit_alpha * (x - fit.logit_beta)))),
          type: "scatter", mode: "lines",
          name: `${r} logit`, showlegend: false,
          line: { color: c, width: 1.2, dash: "longdash" },
          hovertemplate: `<b>${r} logit</b><br>%{x:.2f}: %{y:.3f}<extra></extra>`,
        });
      }
    });
  }

  // ── Plotly layout ──────────────────────────────────────────────────────
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font:   { color: "#e2e8f0", family: "Inter, sans-serif", size: 12 },
    xaxis:  {
      title: { text: labels.x, standoff: 8 },
      gridcolor: "#1a1f35", zerolinecolor: "#1a1f35", color: "#94a3b8",
    },
    yaxis:  {
      title: { text: labels.y, standoff: 8 },
      range: [-0.05, 1.05],
      gridcolor: "#1a1f35", zerolinecolor: "#1a1f35", color: "#94a3b8",
    },
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      bordercolor: "#252d48",
      font: { size: 11 },
    },
    margin:    { t: 14, r: 18, b: 52, l: 60 },
    hovermode: "closest",
  };

  Plotly.react("chart-main", traces, layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d","select2d","autoScale2d"],
    displaylogo: false,
  });

  $("chart-caption").textContent =
    `${cap(S.data.product)}  ·  showing ${regions.length} / ${S.data.regions.length} regions`;
}

// ── Fit results table ──────────────────────────────────────────────────────
function renderFitTable() {
  if (!S.data) return;

  document.querySelector("#fit-table thead").innerHTML = `<tr>
    <th>Region</th>
    <th>Best</th>
    <th>Linear RMSE</th><th>a</th><th>b</th>
    <th>Exp RMSE</th><th>α</th><th>β</th>
    <th>Logit RMSE</th><th>α</th><th>β</th>
  </tr>`;

  const tbody = document.querySelector("#fit-table tbody");
  tbody.innerHTML = "";

  S.data.fit_results.forEach(row => {
    const best = row.best;
    const isP  = best === "Poly";
    const isE  = best === "Power";
    const isL  = best === "Logit";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="region-cell">${row.region}</td>
      <td><span class="badge badge-${best.toLowerCase()}">${best}</span></td>
      <td class="${isP ? "best-cell" : ""}">${fmt(row.poly_error)}</td>
      <td class="${isP ? "best-cell" : ""}">${fmt(row.poly_a)}</td>
      <td class="${isP ? "best-cell" : ""}">${fmt(row.poly_b)}</td>
      <td class="${isE ? "best-cell" : ""}">${fmt(row.power_error)}</td>
      <td class="${isE ? "best-cell" : ""}">${fmt(row.power_alpha)}</td>
      <td class="${isE ? "best-cell" : ""}">${fmt(row.power_beta)}</td>
      <td class="${isL ? "best-cell" : ""}">${fmt(row.logit_error)}</td>
      <td class="${isL ? "best-cell" : ""}">${fmt(row.logit_alpha)}</td>
      <td class="${isL ? "best-cell" : ""}">${fmt(row.logit_beta)}</td>`;
    tbody.appendChild(tr);
  });
}

// ── Sanity check grid ──────────────────────────────────────────────────────
function renderSanity() {
  if (!S.data) return;
  const grid = $("sanity-grid");
  grid.innerHTML = "";

  Object.entries(S.data.sanity_check).forEach(([key, value]) => {
    const cls = "san-" + value.toLowerCase().replace(/\s+/g, "-");
    const div = document.createElement("div");
    div.className = `san-item ${cls}`;
    div.innerHTML = `
      <span class="san-key">${key}</span>
      <span class="san-val">${value}</span>`;
    grid.appendChild(div);
  });
}

// ── State switcher ─────────────────────────────────────────────────────────
function showState(name) {
  ["loading","empty","content"].forEach(s =>
    $(`state-${s}`).classList.toggle("hidden", s !== name));
}

// ── Event wiring ───────────────────────────────────────────────────────────
$("graph-tabs").addEventListener("click", e => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  S.graphType = tab.dataset.g;

  // Enable "overlay fits" only on ratio tab
  const wrap = $("toggle-fits-wrap");
  wrap.classList.toggle("enabled", S.graphType === "ratio");

  renderChart();
});

$("btn-all").addEventListener("click", () => {
  S.selRegions = new Set(S.data?.regions ?? []);
  document.querySelectorAll("#region-list input").forEach(c => c.checked = true);
  renderChart();
});

$("btn-none").addEventListener("click", () => {
  S.selRegions = new Set();
  document.querySelectorAll("#region-list input").forEach(c => c.checked = false);
  renderChart();
});

$("toggle-observed").addEventListener("change", e => {
  S.showObserved = e.target.checked;
  renderChart();
});

$("toggle-fits").addEventListener("change", e => {
  S.showFits = e.target.checked;
  renderChart();
});

// ── Go ─────────────────────────────────────────────────────────────────────
init();
