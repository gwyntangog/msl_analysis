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
  graphType:    "ratio",
  selRegions:   new Set(),
  showObserved: true,
  showFits:     false,
  fitType:      "poly",
  cuPrice:      null,
  alPrice:      null,
};

let definitionData = {};

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
//  JS MODEL  (mirrors Python utility / logit functions)
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
  return (hi - cost) / (hi - lo);
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
//  LIVE CURVE COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/** Generates an evenly-spaced array from start to stop (inclusive) by step. */
function linspace(start, stop, step) {
  const arr = [];
  for (let v = start; v <= stop + step * 1e-9; v += step)
    arr.push(Math.round(v * 1e6) / 1e6);
  return arr;
}

// X-axis domains — mirror the ranges used in export_data.py
const X_RANGES = {
  copper_price:   linspace(100, 20000, 100),   // $/tonne
  aluminum_price: linspace(100, 20000, 50),    // $/tonne
  ratio:          linspace(2.0, 20.0,  0.1),   // dimensionless
};

/**
 * Given region params and current prices ($/tonne), return a y-array of
 * market shares over the full x-range for the chosen graph type.
 *
 * copper_price tab  → sweep Cu price, hold Al price from slider
 * aluminum_price tab → sweep Al price, hold Cu price from slider
 * ratio tab         → sweep ratio = Cu/Al, hold Al price from slider
 */
function computeCurve(p, gType, cuPrice, alPrice) {
  const cuKg = cuPrice / 1000;
  const alKg = alPrice / 1000;

  return X_RANGES[gType].map(x => {
    let ms;
    if      (gType === "copper_price")   ms = marketShareJS(p, x / 1000, alKg);
    else if (gType === "aluminum_price") ms = marketShareJS(p, cuKg, x / 1000);
    else /* ratio */                     ms = marketShareJS(p, alKg * x, alKg);

    return isFinite(ms) ? Math.max(0, Math.min(1, ms)) : 0.5;
  });
}

function tauCallibrate(utility_cu, utility_al, market_share_cu){
  let mc = Math.max(10 ** -9, Math.min(market_share_cu, 1 - 10 ** -9));
  if (mc == 0.5){
      return Infinity;
  }
  else {
      const tau = (utility_cu-utility_al)/(Math.log(mc)-Math.log(1-mc));
      return tau;
  }
  }

// function computeTimeSeries(p, price_list){

//   let ms_series = [];
//   let tau = p.tau_value ?? 1;
//   // let cuKg =  p[`copper_price_per_kg`];
//   let alKg = p[`aluminum_price_per_kg`];
//   let u_cu, u_al, new_ms;
//   for (let price of price_list){
//     u_cu = utilityJS(p, price, alKg, "cu", 5);
//     u_al = utilityJS(p, price, alKg, "al", 5);
//     new_ms = logitMS(u_cu, u_al, tau);
//     ms_series.push(new_ms);
//     tau = tauCallibrate(u_cu, u_al, new_ms);
//   }
//   return ms_series;
// }

function computeTimeSeries(p, priceList, variable = "cu", numAttributes = 5) {
  if (variable !== "cu" && variable !== "al") {
    throw new Error(`variable must be "cu" or "al", got ${variable}`);
  }

  if (!priceList || priceList.length === 0) {
    throw new Error("priceList must contain at least one price.");
  }

  // Don't mutate S.data.region_params
  const row = { ...p };

  let tau = Number(row.tau_value);
  const msSeries = [];

  function utilities(price) {
    if (variable === "cu") {
      return [
        utilityJS(row, price, row.aluminum_price_per_kg, "cu", numAttributes),
        utilityJS(row, price, row.aluminum_price_per_kg, "al", numAttributes),
      ];
    }

    return [
      utilityJS(row, row.copper_price_per_kg, price, "cu", numAttributes),
      utilityJS(row, row.copper_price_per_kg, price, "al", numAttributes),
    ];
  }

  for (const price of priceList) {
    const [uCu, uAl] = utilities(price);

    const newMs = logitMS(uCu, uAl, tau);

    msSeries.push(newMs);

    tau = tauCallibrate(uCu, uAl, newMs);

    row.copper_product_market_share = newMs;
  }

  return msSeries;
}

// ═══════════════════════════════════════════════════════════════════════════
//  PRICE EXPLORER  (bar chart)
// ═══════════════════════════════════════════════════════════════════════════

let _listenersReady = false;

function setupPriceListeners() {
  if (_listenersReady) return;
  _listenersReady = true;

  const sCu = $("slider-cu"), iCu = $("input-cu");
  const sAl = $("slider-al"), iAl = $("input-al");

  function clamp(v, mat) {
    const pm = S.data?.price_meta ?? {};
    return Math.max(pm[`${mat}_min`] ?? 0,
                    Math.min(pm[`${mat}_max`] ?? 30000, v));
  }

  // Both renderExplorer and renderChart update together on every price change
  function setCu(v) { S.cuPrice = v; renderChart(); renderExplorer(); }
  function setAl(v) { S.alPrice = v; renderChart(); renderExplorer(); }

  sCu.addEventListener("input",  () => { iCu.value = sCu.value;             setCu(+sCu.value); });
  iCu.addEventListener("change", () => { const v=clamp(+iCu.value,"cu"); sCu.value=iCu.value=v; setCu(v); });

  sAl.addEventListener("input",  () => { iAl.value = sAl.value;             setAl(+sAl.value); });
  iAl.addEventListener("change", () => { const v=clamp(+iAl.value,"al"); sAl.value=iAl.value=v; setAl(v); });

  $("btn-reset-prices").addEventListener("click", () => {
    const pm = S.data?.price_meta;
    if (!pm) return;
    sCu.value = iCu.value = S.cuPrice = pm.cu_default;
    sAl.value = iAl.value = S.alPrice = pm.al_default;
    renderChart();
    renderExplorer();
  });
}

function initPriceExplorer() {
  setupPriceListeners();

  const pm  = S.data.price_meta;
  const sCu = $("slider-cu"), iCu = $("input-cu");
  const sAl = $("slider-al"), iAl = $("input-al");

  for (const el of [sCu, iCu]) {
    el.min = pm.cu_min; el.max = pm.cu_max; el.step = pm.cu_step;
  }
  for (const el of [sAl, iAl]) {
    el.min = pm.al_min; el.max = pm.al_max; el.step = pm.al_step;
  }

  // Sync sliders to prices already set in loadProduct
  sCu.value = iCu.value = S.cuPrice;
  sAl.value = iAl.value = S.alPrice;

  // Observed average annotation
  const rp = S.data.region_params ?? {};
  const mean = arr => arr.length
    ? Math.round(arr.reduce((a,b) => a+b, 0) / arr.length)
    : null;
  const cuObs = mean(Object.values(rp).map(p => p.copper_price_per_kg   * 1000).filter(Boolean));
  const alObs = mean(Object.values(rp).map(p => p.aluminum_price_per_kg * 1000).filter(Boolean));
  $("note-cu").textContent = cuObs ? `avg observed: $${cuObs.toLocaleString()}/t` : "";
  $("note-al").textContent = alObs ? `avg observed: $${alObs.toLocaleString()}/t` : "";

  renderExplorer();
}

// render overview
function renderOverview() {
  const el = $("overview-panel");
  if (!el || !S.data?.product_info) return;

  const pi   = S.data.product_info;
  const attrs = Object.entries(pi.attributes ?? {});
  const totalWeight = attrs.reduce((s, [, a]) => s + a.weight, 0);
    const prodDefinition = definitionData?.[S.data.product] ?? "No description available.";

  // ── Attribute rows ───────────────────────────────────────────────────
  const attrRows = attrs.map(([key, a]) => {
    const pct     = totalWeight > 0 ? (a.weight / totalWeight * 100).toFixed(1) : 0;
    const dirIcon = a.direction === "positive" ? "↑" : a.direction === "negative" ? "↓" : "—";
    const dirCls  = a.direction === "positive" ? "dir-pos" : a.direction === "negative" ? "dir-neg" : "";
    const attName = a.name;
    return `
      <div class="attr-row">
        <span class="attr-name">${attName}</span>
        <span class="attr-dir ${dirCls}">${dirIcon}</span>
        <div class="attr-bar-wrap">
          <div class="attr-bar" style="width:${pct}%"></div>
        </div>
        <span class="attr-pct">${pct}%</span>
      </div>`;
  }).join("");

  el.innerHTML = `
    <div class="overview-products">
      <div class="overview-material cu-side">
        <span class="mat-dot" style="background:#b45309"></span>
        <div>
          <div class="mat-label">Copper product</div>
          <div class="mat-name">${pi.cu_product_name}</div>
        </div>
      </div>
      <div class="overview-vs">vs</div>
      <div class="overview-material al-side">
        <span class="mat-dot" style="background:#6366f1"></span>
        <div>
          <div class="mat-label">Aluminum product</div>
          <div class="mat-name">${pi.al_product_name}</div>
        </div>
      </div>
    </div>

      <div>
        <div class="overview-definition">
          <div class="def-label">Description</div>
          <div class="def-text">${prodDefinition}</div>
        </div>
      </div>
      <br>

    <div class="overview-attrs">
      <div class="attrs-title">Attribute weights</div>
      ${attrRows}
    </div>
  `;
}

// Preview stats removed
//  <div class="overview-stats">
//       <div class="stat-chip">${pi.num_regions} regions</div>
//       ${pi.avg_tau   != null ? `<div class="stat-chip">avg τ = ${pi.avg_tau.toFixed(3)}</div>` : ""}
//       ${pi.avg_observed_ms != null ? `<div class="stat-chip">avg observed share = ${(pi.avg_observed_ms*100).toFixed(1)}%</div>` : ""}
//     </div>

function renderExplorer() {
  if (!$("chart-explorer")) return;
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
      x: regions, y: computed,
      name: label,
      marker: { color: colors, opacity: 0.82 },
      hovertemplate: "<b>%{x}</b><br>Market share: <b>%{y:.3f}</b><extra></extra>",
    },
    {
      type: "scatter", mode: "markers",
      x: regions, y: observed,
      name: "Observed baseline",
      marker: {
        symbol: "diamond", size: 12,
        color: "rgba(0,0,0,0)",
        line: { color: "#94a3b8", width: 2 },
      },
      hovertemplate: "<b>%{x}</b><br>Observed: <b>%{y:.3f}</b><extra></extra>",
    },
  ];

  const layout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: { color: "#1e293b", family: "Inter, sans-serif", size: 12 }, // 👈 dark
  xaxis: {
    color: "#475569", gridcolor: "rgba(0,0,0,0)",                    // 👈 dark
    tickangle: regions.length > 5 ? -30 : 0,
  },
  yaxis: {
    title: { text: "Cu Market Share", standoff: 8 },
    range: [-0.05, 1.1],
    gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569", // 👈 light grid, dark labels
  },
  shapes: [{
    type: "line",
    x0: -0.5, x1: Math.max(0, regions.length - 0.5),
    y0: 0.5,  y1: 0.5,
    xref: "x", yref: "y",
    line: { color: "#94a3b8", width: 1.2, dash: "dot" },
  }],
  legend: {
    bgcolor: "rgba(0,0,0,0)", bordercolor: "#cbd5e1",
    font: { size: 10, color: "#1e293b" }, orientation: "h", y: -0.32, // 👈 dark
  },
  bargap: 0.38,
  margin: { t: 10, r: 18, b: 85, l: 55 },
  hovermode: "closest",
};

  // Plotly.react("chart-explorer", traces, layout, {
  //   responsive: true, displayModeBar: false,
  // });
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN CHART  — curves computed live from slider prices
// ═══════════════════════════════════════════════════════════════════════════

function renderChart() {
  if (!S.data || S.cuPrice == null || S.alPrice == null) return;

  const gType   = S.graphType;
  const labels  = AXES[gType];
  const regions = S.data.regions.filter(r => S.selRegions.has(r));
  const rp      = S.data.region_params ?? {};
  const traces  = [];
  const xs      = X_RANGES[gType];

  // ── Live curves ────────────────────────────────────────────────────────
  regions.forEach(r => {
    const p = rp[r];
    if (!p) return;
    const c  = colorFor(r);

    // Recompute curve at current slider prices
    const ys = computeCurve(p, gType, S.cuPrice, S.alPrice);

    traces.push({
      x: xs, y: ys,
      type: "scatter", mode: "lines", name: r,
      line: { color: c, width: 2.2 },
      hovertemplate:
        `<b>${r}</b><br>${labels.x}: %{x:.2f}<br>${labels.y}: %{y:.3f}<extra></extra>`,
    });

    // ── Observed point ──────────────────────────────────────────────────
    // if (S.showObserved) {
    //   let obsX;
    //   if      (gType === "copper_price")   obsX = p.copper_price_per_kg   * 1000;
    //   else if (gType === "aluminum_price") obsX = p.aluminum_price_per_kg * 1000;
    //   else /* ratio */                     obsX = p.aluminum_price_per_kg > 0
    //     ? p.copper_price_per_kg / p.aluminum_price_per_kg
    //     : null;

    //   if (obsX != null) {
    //     traces.push({
    //       x: [obsX], y: [p.copper_product_market_share],
    //       type: "scatter", mode: "markers",
    //       name: `${r} (obs)`, showlegend: false,
    //       marker: { color: c, size: 9, symbol: "circle",
    //                 line: { color: "#fff", width: 1.5 } },
    //       hovertemplate:
    //         `<b>${r} — observed</b><br>${labels.x}: %{x:.2f}<br>` +
    //         `${labels.y}: %{y:.3f}<extra></extra>`,
    //     });
    //   }
    // }
  });

  // ── Fit overlays (ratio tab only) ──────────────────────────────────────
  if (S.showFits && gType === "ratio") {
    const { s_min, s_max } = S.data.fit_bounds;
    regions.forEach(r => {
      const fit = S.data.fit_results.find(f => f.region === r);
      if (!fit) return;
      const c = colorFor(r);

      if (fit.poly_a != null)
        traces.push({ x: xs, y: xs.map(x => fit.poly_a + fit.poly_b * x),
          type:"scatter", mode:"lines", showlegend:false, name:`${r} linear`,
          line:{ color:c, width:1.2, dash:"dot" },
          hovertemplate:`<b>${r} linear</b><br>%{x:.2f}: %{y:.3f}<extra></extra>` });

      if (fit.power_alpha != null)
        traces.push({ x: xs, y: xs.map(x => fit.power_alpha * Math.exp(fit.power_beta * x)),
          type:"scatter", mode:"lines", showlegend:false, name:`${r} exp`,
          line:{ color:c, width:1.2, dash:"dash" },
          hovertemplate:`<b>${r} exp</b><br>%{x:.2f}: %{y:.3f}<extra></extra>` });

      if (fit.logit_alpha != null)
        traces.push({ x: xs,
          y: xs.map(x => s_min + (s_max-s_min) / (1 + Math.exp(fit.logit_alpha*(x-fit.logit_beta)))),
          type:"scatter", mode:"lines", showlegend:false, name:`${r} logit`,
          line:{ color:c, width:1.2, dash:"longdash" },
          hovertemplate:`<b>${r} logit</b><br>%{x:.2f}: %{y:.3f}<extra></extra>` });
    });
  }

  // ── Vertical indicator for current price/ratio ─────────────────────────
  const shapes = [], annots = [];

  const addIndicator = (xVal, color, label) => {
    shapes.push({
      type: "line",
      x0: xVal, x1: xVal, y0: 0, y1: 1, yref: "paper",
      line: { color, width: 1.5, dash: "dot" },
    });
    annots.push({
      x: xVal, y: 1.04, yref: "paper",
      text: label, showarrow: false,
      font: { size: 9.5, color },
      xanchor: "center",
      bgcolor: "rgba(11,14,24,.85)",
      borderpad: 2,
    });
  };

  if (gType === "copper_price")
    addIndicator(S.cuPrice, "#f59e0b", `Cu $${S.cuPrice.toLocaleString()}/t`);

  if (gType === "aluminum_price")
    addIndicator(S.alPrice, "#818cf8", `Al $${S.alPrice.toLocaleString()}/t`);

  if (gType === "ratio" && S.alPrice > 0)
    addIndicator(
      +(S.cuPrice / S.alPrice).toFixed(3),
      "#94a3b8",
      `${(S.cuPrice / S.alPrice).toFixed(2)}×`
    );

  // ── Layout ────────────────────────────────────────────────────────────
  const layout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: { color: "#1e293b", family: "Inter, sans-serif", size: 12 },  // 👈 dark
  xaxis: {
    title: { text: labels.x, standoff: 8 },
    gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569", // 👈
  },
  yaxis: {
    title: { text: labels.y, standoff: 8 },
    range: [-0.05, 1.05],
    gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569", // 👈
  },
  shapes, annotations: annots,
  legend: {
    bgcolor: "rgba(0,0,0,0)", bordercolor: "#cbd5e1",
    font: { size: 11, color: "#1e293b" },                              // 👈 dark
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

  //-------------- render fit chart
function renderFitChart() {
  if (!S.data) return;

  const regions  = S.data.regions.filter(r => S.selRegions.has(r));
  const xs       = X_RANGES["ratio"];
  const { s_min, s_max } = S.data.fit_bounds;
  const traces   = [];

  regions.forEach(r => {
    const fit = S.data.fit_results.find(f => f.region === r);
    const p   = S.data.region_params?.[r];
    if (!fit || !p) return;
    const c = colorFor(r);

    // ── Actual computed curve (dashed, for reference) ──────────────────
    const ysActual = computeCurve(p, "ratio", S.cuPrice, S.alPrice);
    traces.push({
      x: xs, y: ysActual,
      type: "scatter", mode: "lines", name: `${r} (model)`,
      opacity: 0.45,
      line: { color: c, width: 2.2 },
      hovertemplate: `<b>${r} model</b><br>Ratio: %{x:.2f}<br>Share: %{y:.3f}<extra></extra>`,
    });

    // ── Fit curve ──────────────────────────────────────────────────────
    let ysFit;
    if (S.fitType === "poly" && fit.poly_a != null) {
      ysFit = xs.map(x => fit.poly_a + fit.poly_b * x);
    } else if (S.fitType === "power" && fit.power_alpha != null) {
      ysFit = xs.map(x => fit.power_alpha * Math.exp(fit.power_beta * x));
    } else if (S.fitType === "logit" && fit.logit_alpha != null) {
      ysFit = xs.map(x =>
        s_min + (s_max - s_min) / (1 + Math.exp(fit.logit_alpha * (x - fit.logit_beta)))
      );
    }

    if (ysFit) {
      traces.push({
        x: xs, y: ysFit,
        type: "scatter", mode: "lines", name: `${r} (${S.fitType} fit)`,
        opacity: 1,
         line: { color: c, width: 1.5, dash: "dot" },
        hovertemplate: `<b>${r} ${S.fitType} fit</b><br>Ratio: %{x:.2f}<br>Share: %{y:.3f}<extra></extra>`,
      });
    }
  });

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { color: "#1e293b", family: "Inter, sans-serif", size: 12 },
    xaxis: {
      title: { text: "Cu / Al Price Ratio", standoff: 8 },
      gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569",
    },
    yaxis: {
      title: { text: "Cu Product Market Share", standoff: 8 },
      range: [-0.05, 1.05],
      gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569",
    },
    legend: {
      bgcolor: "rgba(0,0,0,0)", bordercolor: "#cbd5e1",
      font: { size: 11, color: "#1e293b" },
    },
    margin: { t: 28, r: 18, b: 52, l: 60 },
    hovermode: "closest",
  };

  Plotly.react("chart-fit", traces, layout, {
    responsive: true, displayModeBar: false,
  });

  $("chart-fit-caption").textContent =
    `${cap(S.data.product)}  ·  ${S.fitType} fit vs model curve`;
}

 //-------------- render time chart
function renderTimeChart() {
  if (!S.data) return;

  const regions  = S.data.regions.filter(r => S.selRegions.has(r));
  // const xs       = linspace(10, 100, 100);
  // const { s_min, s_max } = S.data.fit_bounds;
  const traces   = [];

  const years = [
      2027, 2028, 2029, 2030, 2031,
      2032, 2033, 2034, 2035
    ];

  const copperPrices = [
    9.90, 10.08, 10.21, 10.30, 10.41,
    10.52, 10.63, 10.74, 10.85
  ];

  regions.forEach(r => {
    const fit = S.data.fit_results.find(f => f.region === r);
    const p   = S.data.region_params?.[r];
    if (!fit || !p) return;
    const c = colorFor(r);

    // ── Actual computed curve (dashed, for reference) ──────────────────
    const xs = copperPrices;
    const ysActual = computeTimeSeries(p, xs, "cu");
    // console.log(xs);
    // const ysActual = computeTimeSeries(p, xs);
    traces.push({
      x: years,
      y: ysActual,
      customdata: copperPrices,
      type: "scatter",
      mode: "lines",
      name: `${r} (model)`,
      opacity: 0.45,
      line: { color: c, width: 2.2 },
      hovertemplate:
        `<b>${r} model</b>` +
        `<br>Year: %{x}` +
        `<br>Copper price: $%{customdata:.2f}/kg` +
        `<br>Share: %{y:.3f}` +
        `<extra></extra>`,
    });
    // traces.push({
    //   x: years, y: ysActual,
    //   type: "scatter", mode: "lines", name: `${r} (model)`,
    //   opacity: 0.45,
    //   line: { color: c, width: 2.2 },
    //   hovertemplate: `<b>${r} model</b><br>Year: %{x:.2f}<br>Share: %{y:.3f}<extra></extra>`,
    // });
  }
);

//  traces.push({
//       x: years, y: copperPrices,
//       type: "scatter", mode: "lines", name: `Price`,
//       opacity: 0.45,
//       line: { color: "black", width: 2.2 },
//       hovertemplate: `<br>Year: %{x:.2f}<br>Copper Price: %{y:.3f}<extra></extra>`,
//     });

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { color: "#1e293b", family: "Inter, sans-serif", size: 12 },
    xaxis: {
      title: { text: "Year", standoff: 8 },
      gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569",
    },
    yaxis: {
      title: { text: "Cu Product Market Share", standoff: 8 },
      range: [-0.05, 1.05],
      gridcolor: "#e2e8f0", zerolinecolor: "#cbd5e1", color: "#475569",
    },
    legend: {
      bgcolor: "rgba(0,0,0,0)", bordercolor: "#cbd5e1",
      font: { size: 11, color: "#1e293b" },
    },
    margin: { t: 28, r: 18, b: 52, l: 60 },
    hovermode: "closest",
  };

  Plotly.react("chart-time", traces, layout, {
    responsive: true, displayModeBar: false,
  });

  $("chart-time-caption").textContent =
    `${cap(S.data.product)}  ·  market share prediction over time`;
}






// ═══════════════════════════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════════════════════════

async function init() {

  try {
     // ── Load definitions once, independently ──────────────────────────
    try {
      definitionData = await fetchJSON("manual_data/definitions.json");
    } catch (e) {
      console.warn("definitions.json not found — descriptions will be blank.", e.message);
      definitionData = {};          // graceful fallback; rest of app still works
    }
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
    const btn       = document.createElement("button");
    btn.className   = "prod-btn";
    btn.textContent = cap(p);
    btn.dataset.p   = p;
    btn.addEventListener("click", () => loadProduct(p));
    el.appendChild(btn);
  });
}


// ── Data table ─────────────────────────────────────────────────────────────
let _dfListenersReady = false;

function setupDataTableListeners() {
  if (_dfListenersReady) return;

  // const search    = $("df-search");
  const exportBtn = $("btn-export-df");

  if (!exportBtn) {
    console.warn("Data table elements not found in DOM — check IDs in HTML");
    return;
  }

  // search.addEventListener("input", e => renderDataTable(e.target.value));

  exportBtn.addEventListener("click", () => {
    if (!S.data?.dataframe?.length) return;
    const cols = Object.keys(S.data.dataframe[0]);
    const csv  = [
      cols.join(","),
      ...S.data.dataframe.map(row =>
        cols.map(c => {
          const v = row[c] ?? "";
          return typeof v === "string" ? `"${v}"` : v;
        }).join(",")
      ),
    ].join("\n");

    const a = Object.assign(document.createElement("a"), {
      href:     URL.createObjectURL(new Blob([csv], { type: "text/csv" })),
      download: `${S.data.product}_data.csv`,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  });

  _dfListenersReady = true;
}

function renderDataTable() {
  if (!S.data?.dataframe?.length) return;

  const rows = S.data.dataframe;
  const cols = Object.keys(rows[0]);

  const thead = document.querySelector("#df-table thead");
  if (!thead.innerHTML) {
    thead.innerHTML = `<tr>${cols.map(c =>
      `<th title="${c}">${cap(c)}</th>`
    ).join("")}</tr>`;
  }

  const tbody = document.querySelector("#df-table tbody");
  tbody.innerHTML = rows.map(row =>
    `<tr>${cols.map(c => {
      const cls = c === "region" ? " class=\"region-cell\"" : "";
      return `<td${cls}>${fmt(row[c])}</td>`;
    }).join("")}</tr>`
  ).join("");

  $("df-caption").textContent =
    `${rows.length} rows · ${cols.length} columns`;
}

// PRODUCT

async function loadProduct(product) {
  document.querySelectorAll(".prod-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.p === product));

  showState("loading");
  try {
    S.data       = await fetchJSON(`data/${product}.json`);
    document.querySelector("#df-table thead").innerHTML = "";
    // $("df-search").value = "";
    S.selRegions = new Set(S.data.regions);

    // Set prices BEFORE renderAll so computeCurve has values to work with
    const pm = S.data.price_meta;
    S.cuPrice = pm.cu_default;
    S.alPrice = pm.al_default;

    buildRegionList(S.data.regions);
    $("header-subtitle").textContent =
      `${cap(product)}  ·  ${S.data.regions.length} regions`;
    renderAll();
    initPriceExplorer();
    showState("content");
    setupDataTableListeners();
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
    const label     = document.createElement("label");
    label.className = "reg-item";
    label.innerHTML = `
      <input type="checkbox" value="${r}" checked />
      <span class="reg-dot" style="background:${colorFor(r)}"></span>
      ${r}`;
    label.querySelector("input").addEventListener("change", e => {
      e.target.checked ? S.selRegions.add(r) : S.selRegions.delete(r);
      renderChart();
      renderExplorer();
      renderFitChart();
      renderTimeChart();
    });
    el.appendChild(label);
  });
}

function renderAll() {
  renderOverview();
  const isRatio = S.graphType === "ratio";
  $("card-explorer").classList.toggle("hidden", S.graphType === "ratio");
  $("card-fit-curves").classList.toggle("hidden", !isRatio);
  renderChart();
  renderFitChart();
  renderFitTable();
  renderTimeChart();
  renderSanity();
  renderDataTable();
}

// ── Fit table ──────────────────────────────────────────────────────────────
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
    const b  = row.best;
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

function exportFitResults() {
  if (!S.data || !S.data.fit_results || S.data.fit_results.length === 0) {
    alert("No fit results to export.");
    return;
  }

  const rows = S.data.fit_results;
  const headers = Object.keys(rows[0]);

  const csv = [
    headers.join(","),
    ...rows.map(row =>
      headers.map(h => {
        const val = row[h];
        if (val === null || val === undefined) return "";
        // Wrap strings in quotes in case they contain commas
        if (typeof val === "string") return `"${val}"`;
        return val;
      }).join(",")
    )
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${S.data.product}_fit_results.csv`;
  a.click();
  URL.revokeObjectURL(url);
}


function renderSanity() {
  const grid = document.getElementById("sanity-grid");
  if (!grid) return;
  grid.innerHTML = "";

  if (!S.data) {
    grid.innerHTML = `<div class="sanity-card sanity-error">No data</div>`;
    return;
  }

  const text = S.data.sanity_check;

  if (!text || text.trim() === "") {
    grid.innerHTML = `<div class="sanity-card sanity-ok">✓ All Clear</div>`;
    return;
  }

  const lines = text.split("\n").map(l => l.trim()).filter(l => l !== "");
  lines.forEach(line => {
    const card = document.createElement("div");
    card.className = "sanity-card sanity-error";
    card.textContent = "⚠ " + line;
    grid.appendChild(card);
  });
}

// ── State & events ─────────────────────────────────────────────────────────
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
  // $("toggle-fits-wrap").classList.toggle("enabled", S.graphType === "ratio");
  $("card-explorer").classList.toggle("hidden", S.graphType === "ratio");
  $("card-fit-curves").classList.toggle("hidden", S.graphType !== "ratio");
  renderChart();
  renderFitChart();
});

$("fit-tabs").addEventListener("click", e => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  document.querySelectorAll("#fit-tabs .tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  S.fitType = tab.dataset.f;
  renderFitChart();
});

$("btn-all").addEventListener("click", () => {
  S.selRegions = new Set(S.data?.regions ?? []);
  document.querySelectorAll("#region-list input").forEach(c => c.checked = true);
  renderChart(); renderExplorer(); renderFitChart(); renderTimeChart();
});

$("btn-none").addEventListener("click", () => {
  S.selRegions = new Set();
  document.querySelectorAll("#region-list input").forEach(c => c.checked = false);
  renderChart(); renderExplorer(); renderFitChart(); renderTimeChart();
});

$("toggle-observed").addEventListener("change", e => {
  S.showObserved = e.target.checked;
  renderChart();
});



// $("toggle-fits").addEventListener("change", e => {
//   S.showFits = e.target.checked;
//   renderChart();
// });

init();
