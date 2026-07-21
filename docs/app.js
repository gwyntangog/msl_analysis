/* app.js ─────────────────────────────────────────────────────────────────────
 * Material Substitution Model — interactive dashboard
 * ─────────────────────────────────────────────────────────────────────────── */
"use strict";

// ── Palette & constants ────────────────────────────────────────────────────
const PALETTE = [
  "#3b82f6","#10b981","#f59e0b","#ef4444","#a78bfa",
  "#06b6d4","#f97316","#84cc16","#ec4899","#6366f1",
];

const AXES = {
  copper_price:   { x: "Copper Price ($/tonne)",   y: "Cu Product Market Share" },
  aluminum_price: { x: "Aluminum Price ($/tonne)", y: "Cu Product Market Share" },
  ratio:          { x: "Cu / Al Price Ratio",       y: "Cu Product Market Share" },
};

// ── Shared state ───────────────────────────────────────────────────────────
const S = {
  data:         null,
  graphType:    "copper_price",
  selRegions:   new Set(),
  showObserved: true,
  showFits:     false,
  cuPrice:      null,   // $/tonne  (model internally uses $/kg = this / 1000)
  alPrice:      null,
};

// ── Helpers ────────────────────────────────────────────────────────────────
const $   = id  => document.getElementById(id);
const fmt = v   => (v == null ? "—" : typeof v === "number" ? v.toFixed(4) : String(v));
const cap = str => str.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
const colorFor = region => {
  const idx = (S.data?.regions ?? []).indexOf(region);
  return PALETTE[idx % PALETTE.length];
};

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} — ${url}`);
  return r.json();
}

// ═══════════════════════════════════════════════════════════════════════════
//  JS MODEL  (mirrors Python: calc_product_cost_row → normalize → calc_utility_row → ms_logit)
//  All price arguments are in $/kg (the model's native unit).
// ═══════════════════════════════════════════════════════════════════════════

function productCostJS(p, cuKg, alKg, material) {
  return (p[`${material}_non_material_cost_per_unit`] ?? 0)
       + (p[`${material}_copper_kg`]                  ?? 0) * cuKg
       + (p[`${material}_aluminum_kg`]                ?? 0) * alKg;
}

function normCostJS(p, cost) {
  const hi = p.attribute_1_max ?? 1;
  const lo = p.attribute_1_min ?? 0;
  if (hi === lo) return 0;
  return (hi - cost) / (hi - lo);   // negative direction: lower cost → higher score
}

function utilityJS(p, cuKg, alKg, material, n = 5) {
  const cost = productCostJS(p, cuKg, alKg, material);
  let u = normCostJS(p, cost) * (p.weight_attribute_1 ?? 0);
  for (let i = 2; i <= n; i++) {
    u += (p[`${material}_a${i}_callibrated`] ?? 0)
       * (p[`weight_attribute_${i}`]          ?? 0);
  }
  return u;
}

// Numerically stable logit (avoids overflow for large utility differences)
function logitMS(cuU, alU, tau) {
  if (!isFinite(tau) || tau === 0) return 0.5;
  const d = (cuU - alU) / tau;
  if (d >  700) return 1;
  if (d < -700) return 0;
  return 1 / (1 + Math.exp(-d));
}

function marketShareJS(p, cuKg, alKg) {
  return logitMS(
    utilityJS(p, cuKg, alKg, "cu"),
    utilityJS(p, cuKg, alKg, "al"),
    p.tau_value ?? 1
  );
}

// ═══════════════════════════════════════════════════════════════════════════
//  PRICE EXPLORER
// ═══════════════════════════════════════════════════════════════════════════

let _priceListenersReady = false;

/** Wire up the slider ↔ number-input ↔ state sync (runs once). */
function setupPriceListeners() {
  if (_priceListenersReady) return;
  _priceListenersReady = true;

  const sCu = $("slider-cu"), iCu = $("input-cu");
  const sAl = $("slider-al"), iAl = $("input-al");

  function clamp(v, mat) {
    const pm = S.data?.price_meta ?? {};
    return Math.max(pm[`${mat}_min`] ?? 0,
                    Math.min(pm[`${mat}_max`] ?? 30000, v));
  }

  function setCu(v) { S.cuPrice = v; renderExplorer(); renderChart(); }
  function setAl(v) { S.alPrice = v; renderExplorer(); renderChart(); }

  sCu.addEventListener("input",  () => { iCu.value = sCu.value;            setCu(+sCu.value); });
  iCu.addEventListener("change", () => { const v = clamp(+iCu.value,"cu"); sCu.value = iCu.value = v; setCu(v); });

  sAl.addEventListener("input",  () => { iAl.value = sAl.value;            setAl(+sAl.value); });
  iAl.addEventListener("change", () => { const v = clamp(+iAl.value,"al"); sAl.value = iAl.value = v; setAl(v); });

  $("btn-reset-prices").addEventListener("click", () => {
    const pm = S.data?.price_meta;
    if (!pm) return;
    sCu.value = iCu.value = S.cuPrice = pm.cu_default;
    sAl.value = iAl.value = S.alPrice = pm.al_default;
    renderExplorer();
    renderChart();
  });
}

/** Called on every product load — resets slider bounds, defaults, and notes. */
function initPriceExplorer() {
  setupPriceListeners();

  const pm  = S.data.price_meta;
  const sCu = $("slider-cu"), iCu = $("input-cu");
  const sAl = $("slider-al"), iAl = $("input-al");

  // Apply slider config for this product
  for (const el of [sCu, iCu]) {
    el.min = pm.cu_min; el.max = pm.cu_max; el.step = pm.cu_step;
  }
  for (const el of [sAl, iAl]) {
    el.min = pm.al_min; el.max = pm.al_max; el.step = pm.al_step;
  }

  // Reset to defaults
  sCu.value = iCu.value = S.cuPrice = pm.cu_default;
  sAl.value = iAl.value = S.alPrice = pm.al_default;

  // Observed average annotation
  const rp = S.data.region_params ?? {};
  const mean = arr => arr.length
    ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length)
    : null;

  const cuObs = mean(Object.values(rp)
    .map(p => p.copper_price_per_kg * 1000).filter(Boolean));
  const alObs = mean(Object.values(rp)
    .map(p => p.aluminum_price_per_kg * 1000).filter(Boolean));

  $("note-cu").textContent = cuObs ? `avg observed: $${cuObs.toLocaleString()}/t` : "";
  $("note-al").textContent = alObs ? `avg observed: $${alObs.toLocaleString()}/t` : "";

  renderExplorer();
}

/** Renders the bar chart in the Price Explorer card. */
function renderExplorer() {
  if (!S.data?.region_params || S.cuPrice == null) return;

  const regions = S.data.regions.filter(r => S.selRegions.has(r));
  const cuKg    = S.cuPrice / 1000;
  const alKg    = S.alPrice / 1000;
  const rp      = S.data.region_params;

  const computed = regions.map(r => {
    const p = rp[r];
    return p ? +Math.max(0, Math.min(1, marketShareJS(p, cuKg, alKg))).toFixed(4) : null;
  });
  const observed = regions.map(r => rp[r]?.copper_product_market_share ?? null);
  const colors   = regions.map(r => colorFor(r));

  const label = `Cu $${S.cuPrice.toLocaleString()}/t · Al $${S.alPrice.toLocaleString()}/t`;

  const traces = [
    {
      type: "bar",
      x: regions,
      y: computed,
      name: label,
      marker: { color: colors, opacity: 0.82 },
      hovertemplate: "<b>%{x}</b><br>Market share: <b>%{y:.3f}</b><extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      x: regions,
      y: observed,
      name: "Observed baseline",
      marker: {
        symbol: "diamond",
        size: 12,
        color: "rgba(0,0,0,0)",
        line: { color: "#94a3b8", width: 2 },
      },
      hovertemplate: "<b>%{x}</b><br>Observed: <b>%{y:.3f}</b><extra></extra>",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { color: "#e2e8f0", family: "Inter, sans-serif", size: 12 },
    xaxis: {
      color: "#94a3b8",
      gridcolor: "rgba(0,0,0,0)",
      tickangle: regions.length > 5 ? -30 : 0,
    },
    yaxis: {
      title: { text: "Cu Market Share", standoff: 8 },
      range: [-0.05, 1.1],
      gridcolor: "#1a1f35",
      zerolinecolor: "#1a1f35",
      color: "#94a3b8",
    },
    // 50 % reference line
    shapes: [{
      type: "line",
      x0: -0.5, x1: Math.max(0, regions.length - 0.5),
      y0: 0.5,  y1: 0.5,
      xref: "x", yref: "y",
      line: { color: "#334155", width: 1.2, dash: "dot" },
    }],
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      bordercolor: "#252d48",
      font: { size: 10 },
      orientation: "h",
      y: -0.32,
    },
    bargap:    0.38,
    margin:    { t: 10, r: 18, b: 85, l: 55 },
    hovermode: "closest",
  };

  Plotly.react("chart-explorer", traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════════════

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
    initPriceExplorer();
    showState("content");
  } catch (e) {
    console.error(e);
    $("state-empty").innerHTML =
      `<p class="muted">Failed to load <code>${product}.json</code>: ${e.message}</p>`;
    showState("empty");
  }
}

function buildRegionList(regions) {
  const el = $("region-list");
  el.innerHTML = "";
  regions.forEach(r => {
    const label = document.createElement("label");
    label.className = "reg-item";
    label.innerHTML = `
      <input type="checkbox" value="${r}" checked />
      <span class="reg-dot" style="background:${colorFor(r)}"></span>
      ${r}`;
    label.querySelector("input").addEventListener("change", e => {
      e.target.checked ? S.selRegions.add(r) : S.selRegions.delete(r);
      renderChart();
      renderExplorer();
    });
    el.appendChild(label);
  });
}

function renderAll() {
  renderChart();
  renderFitTable();
  renderSanity();
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN PRICE SWEEP CHART  (with sliding vertical price indicator)
// ═══════════════════════════════════════════════════════════════════════════

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

    if (S.showObserved && d.observed_x != null) {
      traces.push({
        x: [d.observed_x], y: [d.observed_y],
        type: "scatter", mode: "markers",
        name: `${r} (obs)`, showlegend: false,
        marker: { color: c, size: 9, symbol: "circle",
                  line: { color: "#fff", width: 1.5 } },
        hovertemplate:
          `<b>${r} — observed</b><br>${labels.x}: %{x:.1f}<br>` +
          `${labels.y}: %{y:.3f}<extra></extra>`,
      });
    }
  });

  // ── Fit overlays (ratio tab only) ──────────────────────────────────────
  if (S.showFits && gType === "ratio") {
    const { s_min, s_max } = S.data.fit_bounds;
    regions.forEach(r => {
      const fit = S.data.fit_results.find(f => f.region === r);
      if (!fit) return;
      const xs = gData[r].x;
      const c  = colorFor(r);

      if (fit.poly_a != null)
        traces.push({ x: xs, y: xs.map(x => fit.poly_a + fit.poly_b * x),
          type: "scatter", mode: "lines", showlegend: false,
          name: `${r} linear`, line: { color: c, width: 1.2, dash: "dot" },
          hovertemplate: `<b>${r} linear</b><br>%{x:.2f}: %{y:.3f}<extra></extra>` });

      if (fit.power_alpha != null)
        traces.push({ x: xs, y: xs.map(x => fit.power_alpha * Math.exp(fit.power_beta * x)),
          type: "scatter", mode: "lines", showlegend: false,
          name: `${r} exp`, line: { color: c, width: 1.2, dash: "dash" },
          hovertemplate: `<b>${r} exp</b><br>%{x:.2f}: %{y:.3f}<extra></extra>` });

      if (fit.logit_alpha != null)
        traces.push({ x: xs, y: xs.map(x =>
          s_min + (s_max - s_min) / (1 + Math.exp(fit.logit_alpha * (x - fit.logit_beta)))),
          type: "scatter", mode: "lines", showlegend: false,
          name: `${r} logit`, line: { color: c, width: 1.2, dash: "longdash" },
          hovertemplate: `<b>${r} logit</b><br>%{x:.2f}: %{y:.3f}<extra></extra>` });
    });
  }

  // ── Vertical price indicator from slider ──────────────────────────────
  const shapes = [], annots = [];

  const addIndicator = (price, color, label) => {
    shapes.push({
      type: "line",
      x0: price, x1: price, y0: 0, y1: 1, yref: "paper",
      line: { color, width: 1.5, dash: "dot" },
    });
    annots.push({
      x: price, y: 1.04, yref: "paper",
      text: label,
      showarrow: false,
      font: { size: 9.5, color },
      xanchor: "center",
      bgcolor: "rgba(11,14,24,.8)",
      borderpad: 2,
    });
  };

  if (gType === "copper_price"   && S.cuPrice != null)
    addIndicator(S.cuPrice, "#f59e0b", `Cu $${S.cuPrice.toLocaleString()}/t`);
  if (gType === "aluminum_price" && S.alPrice != null)
    addIndicator(S.alPrice, "#818cf8", `Al $${S.alPrice.toLocaleString()}/t`);

  // ── Layout ────────────────────────────────────────────────────────────
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
    shapes,
    annotations: annots,
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      bordercolor: "#252d48",
      font: { size: 11 },
    },
    margin:    { t: 28, r: 18, b: 52, l: 60 },
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

// ═══════════════════════════════════════════════════════════════════════════
//  FIT TABLE & SANITY CHECK
// ═══════════════════════════════════════════════════════════════════════════

function renderFitTable() {
  if (!S.data) return;
  document.querySelector("#fit-table thead").innerHTML = `<tr>
    <th>Region</th><th>Best</th>
    <th>Linear RMSE</th><th>a</th><th>b</th>
    <th>Exp RMSE</th><th>α</th><th>β</th>
    <th>Logit RMSE</th><th>α</th><th>β</th>
  </tr>`;
  const tbody = document.querySelector("#fit-table tbody");
  tbody.innerHTML = "";
  S.data.fit_results.forEach(row => {
    const b = row.best;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="region-cell">${row.region}</td>
      <td><span class="badge badge-${b.toLowerCase()}">${b}</span></td>
      <td class="${b==="Poly" ?"best-cell":""}">${fmt(row.poly_error)}</td>
      <td class="${b==="Poly" ?"best-cell":""}">${fmt(row.poly_a)}</td>
      <td class="${b==="Poly" ?"best-cell":""}">${fmt(row.poly_b)}</td>
      <td class="${b==="Power"?"best-cell":""}">${fmt(row.power_error)}</td>
      <td class="${b==="Power"?"best-cell":""}">${fmt(row.power_alpha)}</td>
      <td class="${b==="Power"?"best-cell":""}">${fmt(row.power_beta)}</td>
      <td class="${b==="Logit"?"best-cell":""}">${fmt(row.logit_error)}</td>
      <td class="${b==="Logit"?"best-cell":""}">${fmt(row.logit_alpha)}</td>
      <td class="${b==="Logit"?"best-cell":""}">${fmt(row.logit_beta)}</td>`;
    tbody.appendChild(tr);
  });
}

function renderSanity() {
  if (!S.data) return;
  const grid = $("sanity-grid");
  grid.innerHTML = "";
  Object.entries(S.data.sanity_check).forEach(([key, value]) => {
    const cls = "san-" + value.toLowerCase().replace(/\s+/g, "-");
    const div = document.createElement("div");
    div.className = `san-item ${cls}`;
    div.innerHTML =
      `<span class="san-key">${key}</span><span class="san-val">${value}</span>`;
    grid.appendChild(div);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
//  SHARED EVENT WIRING
// ═══════════════════════════════════════════════════════════════════════════

function showState(name) {
  ["loading","empty","content"].forEach(s =>
    $(`state-${s}`).classList.toggle("hidden", s !== name));
}

$("graph-tabs").addEventListener("click", e => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  S.graphType = tab.dataset.g;
  $("toggle-fits-wrap").classList.toggle("enabled", S.graphType === "ratio");
  renderChart();
});

$("btn-all").addEventListener("click", () => {
  S.selRegions = new Set(S.data?.regions ?? []);
  document.querySelectorAll("#region-list input").forEach(c => c.checked = true);
  renderChart();
  renderExplorer();
});

$("btn-none").addEventListener("click", () => {
  S.selRegions = new Set();
  document.querySelectorAll("#region-list input").forEach(c => c.checked = false);
  renderChart();
  renderExplorer();
});

$("toggle-observed").addEventListener("change", e => {
  S.showObserved = e.target.checked;
  renderChart();
});

$("toggle-fits").addEventListener("change", e => {
  S.showFits = e.target.checked;
  renderChart();
});

init();
