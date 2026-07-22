/* HALLMARK companion site.
   All dynamic content renders from window.HALLMARK_DATA (site_data.js, generated
   by scripts/generate_site_data.py). Data strings enter the DOM via textContent
   only. Charts are hand-rolled SVG/CSS following a validated dataviz method:
   one axis, thin marks, 2px surface gaps/rings, tooltips that enhance but never
   gate (every chart has a table view). */

(function () {
  "use strict";

  var D = window.HALLMARK_DATA;
  if (!D) {
    document.getElementById("kpi-row").textContent =
      "Data failed to load - site_data.js missing.";
    return;
  }

  /* ---------- tiny DOM helpers (textContent only for data strings) ---------- */

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "text") node.textContent = attrs[k];
        else if (k === "style") node.setAttribute("style", attrs[k]);
        else if (k === "role" || k.indexOf("data-") === 0 || k.indexOf("aria-") === 0) {
          node.setAttribute(k, attrs[k]);
        } else node[k] = attrs[k];
      });
    }
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function svgEl(tag, attrs, children) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "text") node.textContent = attrs[k];
        else node.setAttribute(k, attrs[k]);
      });
    }
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function fmt(x, digits) {
    if (x === null || x === undefined || (typeof x === "number" && isNaN(x))) return "–";
    return Number(x).toFixed(digits === undefined ? 3 : digits);
  }

  function pct(x, digits) {
    if (x === null || x === undefined) return "–";
    return (100 * x).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  /* ---------- theme ---------- */

  (function themeSetup() {
    var btn = document.getElementById("theme-toggle");
    var stored = null;
    try { stored = localStorage.getItem("hallmark-theme"); } catch (e) { /* private mode */ }
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
    }
    function effective() {
      var t = document.documentElement.getAttribute("data-theme");
      if (t) return t;
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    function setLabel() {
      btn.textContent = effective() === "dark" ? "☀ Light" : "☾ Dark";
    }
    btn.addEventListener("click", function () {
      var next = effective() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("hallmark-theme", next); } catch (e) { /* ignore */ }
      setLabel();
      rerenderThemed();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      setLabel();
      rerenderThemed();
    });
    setLabel();
  })();

  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  /* ---------- tooltip (one instance; enhances, never gates) ---------- */

  var tooltip = document.getElementById("viz-tooltip");

  function tipShow(evt, title, rows) {
    clear(tooltip);
    if (title) tooltip.appendChild(el("div", { className: "tt-title", text: title }));
    (rows || []).forEach(function (r) {
      var row = el("div", { className: "tt-row" });
      if (r.color) {
        row.appendChild(el("span", { className: "tt-key", style: "background:" + r.color }));
      }
      row.appendChild(el("b", { text: r.value }));
      row.appendChild(document.createTextNode(" " + r.label));
      tooltip.appendChild(row);
    });
    tooltip.style.display = "block";
    tipMove(evt);
  }

  function tipMove(evt) {
    var pad = 14;
    var w = tooltip.offsetWidth, h = tooltip.offsetHeight;
    var x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }

  function tipHide() { tooltip.style.display = "none"; }

  function addTip(node, titleFn, rowsFn) {
    node.addEventListener("pointerenter", function (e) { tipShow(e, titleFn(), rowsFn()); });
    node.addEventListener("pointermove", tipMove);
    node.addEventListener("pointerleave", tipHide);
    node.addEventListener("focus", function () {
      var r = node.getBoundingClientRect();
      tipShow({ clientX: r.right, clientY: r.top }, titleFn(), rowsFn());
    });
    node.addEventListener("blur", tipHide);
  }

  /* ---------- palette (mirrors style.css; sequential ramp indexed, not eyeballed) ---------- */

  var SEQ_LIGHT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

  function seqColor(v) {
    // near-zero recedes toward the surface in both modes
    var ramp = isDark() ? SEQ_LIGHT.slice().reverse() : SEQ_LIGHT;
    var idx = Math.max(0, Math.min(ramp.length - 1, Math.round(v * (ramp.length - 1))));
    return ramp[idx];
  }

  var CAT_COLOR = {}; // category key -> css var
  D.categories.forEach(function (c) { CAT_COLOR[c.key] = "var(--series-" + c.slot + ")"; });

  var ORDINAL_TIER_LIGHT = ["#86b6ef", "#2a78d6", "#104281"]; // steps 250/450/650
  var ORDINAL_TIER_DARK = ["#86b6ef", "#3987e5", "#184f95"];  // steps 250/400/600
  function tierColor(tier) { return (isDark() ? ORDINAL_TIER_DARK : ORDINAL_TIER_LIGHT)[tier - 1]; }

  /* ---------- lookups ---------- */

  var MODEL = {};
  D.models.forEach(function (m) { MODEL[m.tool] = m; });

  var CAT_ORDER = {};
  D.categories.forEach(function (c, i) { CAT_ORDER[c.key] = i; });

  function byCategoryThenName(a, b) {
    var ma = MODEL[a.tool], mb = MODEL[b.tool];
    var d = CAT_ORDER[ma.category] - CAT_ORDER[mb.category];
    if (d) return d;
    return ma.name < mb.name ? -1 : ma.name > mb.name ? 1 : 0;
  }

  var TYPE = {};
  D.taxonomy.forEach(function (t) { TYPE[t.key] = t; });

  function modelName(tool) { return MODEL[tool] ? MODEL[tool].name : tool; }

  /* ---------- KPI row ---------- */

  (function renderKpis() {
    var row = document.getElementById("kpi-row");
    D.kpis.forEach(function (k) {
      row.appendChild(el("div", { className: "stat-tile" }, [
        el("div", { className: "label", text: k.label }),
        el("div", { className: "value", text: k.value }),
        k.sub ? el("div", { className: "sub", text: k.sub }) : null,
      ]));
    });
  })();

  /* ---------- failure-mode visuals ---------- */

  function dumbbellChart(rows, opts) {
    // rows: [{name, a, b}] with a -> b; two shades of one hue (before/after)
    var W = 520, rowH = 34, padL = opts.padL || 150, padR = 56, padT = 26, padB = 30;
    var H = padT + rows.length * rowH + padB;
    var xmax = opts.xmax || Math.min(1, Math.max(0.05, Math.max.apply(null, rows.map(function (r) {
      return Math.max(r.a, r.b);
    })) * 1.15));
    function x(v) { return padL + (v / xmax) * (W - padL - padR); }
    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H, width: "100%",
      role: "img", "aria-label": opts.aria || "dumbbell chart",
    });
    // gridlines at clean ticks
    var tickStep = xmax > 0.6 ? 0.25 : xmax > 0.3 ? 0.1 : 0.05;
    for (var tv = 0; tv <= xmax + 1e-9; tv += tickStep) {
      svg.appendChild(svgEl("line", {
        x1: x(tv), x2: x(tv), y1: padT - 8, y2: H - padB + 4, class: "gridline",
      }));
      svg.appendChild(svgEl("text", {
        x: x(tv), y: H - padB + 18, "text-anchor": "middle", class: "ticklab",
        text: Math.round(tv * 100) + "%",
      }));
    }
    rows.forEach(function (r, i) {
      var cy = padT + i * rowH + rowH / 2;
      svg.appendChild(svgEl("text", {
        x: padL - 10, y: cy + 4, "text-anchor": "end", class: "directlab dim", text: r.name,
      }));
      var g = svgEl("g", { tabindex: "0" });
      g.appendChild(svgEl("line", {
        x1: x(r.a), x2: x(r.b), y1: cy, y2: cy,
        stroke: "var(--baseline)", "stroke-width": 2,
      }));
      [["a", "var(--shade-pre)"], ["b", "var(--shade-post)"]].forEach(function (pair) {
        g.appendChild(svgEl("circle", {
          cx: x(r[pair[0]]), cy: cy, r: 6, fill: pair[1],
          stroke: "var(--surface-1)", "stroke-width": 2,
        }));
      });
      // endpoint value labels (sparing: this chart is <= 6 rows)
      g.appendChild(svgEl("text", {
        x: x(Math.max(r.a, r.b)) + 11, y: cy + 4, class: "directlab",
        text: Math.round(r.b * 100) + "%",
      }));
      var hit = svgEl("rect", {
        x: padL, y: cy - rowH / 2, width: W - padL - padR, height: rowH,
        fill: "transparent",
      });
      addTip(hit, function () { return r.name; }, function () {
        return [
          { value: pct(r.a), label: opts.aLabel, color: "var(--shade-pre)" },
          { value: pct(r.b), label: opts.bLabel, color: "var(--shade-post)" },
        ];
      });
      g.appendChild(hit);
      svg.appendChild(g);
    });
    var wrap = el("div", {}, []);
    var legend = el("div", { className: "legend" }, [
      el("span", { className: "item" }, [
        el("span", { className: "swatch", style: "background:var(--shade-pre);border-radius:50%" }),
        opts.aLabel,
      ]),
      el("span", { className: "item" }, [
        el("span", { className: "swatch", style: "background:var(--shade-post);border-radius:50%" }),
        opts.bLabel,
      ]),
    ]);
    wrap.appendChild(legend);
    wrap.appendChild(svg);
    if (opts.note) wrap.appendChild(el("p", { className: "axis-note", text: opts.note }));
    return wrap;
  }

  function barRows(rows, opts) {
    // rows: [{name, value, color?}] horizontal bars, direct value labels
    var max = opts.max || Math.max.apply(null, rows.map(function (r) { return r.value; }));
    var wrap = el("div", { className: "bar-rows" });
    rows.forEach(function (r) {
      var fillColor = r.color || "var(--series-1)";
      var row = el("div", { className: "bar-row" }, [
        el("span", { className: "name", text: r.name }),
        el("div", { className: "track" }, [
          el("div", {
            className: "fill",
            style: "width:" + Math.max(0.5, (r.value / max) * 100) + "%;background:" + fillColor,
          }),
        ]),
        el("span", { className: "val", text: opts.format ? opts.format(r.value) : pct(r.value) }),
      ]);
      addTip(row, function () { return r.name; }, function () {
        return [{ value: opts.format ? opts.format(r.value) : pct(r.value), label: opts.valueLabel || "", color: fillColor }];
      });
      wrap.appendChild(row);
    });
    var outer = el("div", {}, [wrap]);
    if (opts.note) outer.appendChild(el("p", { className: "axis-note", text: opts.note }));
    return outer;
  }

  function renderFailureModes() {
    var fm = D.failure_modes;
    var m1 = document.getElementById("viz-mode1");
    clear(m1);
    m1.appendChild(dumbbellChart(fm.mode1.rows.map(function (r) {
      return { name: r.name, a: r.zero_shot_fpr, b: r.agentic_fpr };
    }), {
      aLabel: "zero-shot FPR", bLabel: "agentic FPR", padL: 152,
      aria: "False-positive rate, zero-shot versus agentic, per model",
      note: fm.mode1.note,
    }));

    var m2 = document.getElementById("viz-mode2");
    clear(m2);
    m2.appendChild(barRows(fm.mode2.rows.map(function (r) {
      return { name: r.name, value: r.ppv };
    }), {
      valueLabel: "precision at a 2% base rate",
      note: fm.mode2.note,
    }));

    var m3 = document.getElementById("viz-mode3");
    clear(m3);
    m3.appendChild(dumbbellChart(fm.mode3.rows.map(function (r) {
      return { name: r.name, a: r.fpr_indist, b: r.fpr_post };
    }), {
      aLabel: "FPR, 2021–23 corpus", bLabel: "FPR, 2024–25 supplement", padL: 118,
      aria: "False-positive rate before and after training cutoff, per model",
      note: fm.mode3.note,
    }));
  }

  /* ---------- benchmark section ---------- */

  var taxSelected = null;

  function renderTaxonomy() {
    var grid = document.getElementById("tax-grid");
    clear(grid);
    D.taxonomy.forEach(function (t) {
      var card = el("div", {
        className: "tax-card" + (taxSelected === t.key ? " on" : ""),
        tabIndex: 0, role: "button",
      }, [
        el("h4", {}, [
          document.createTextNode(t.name),
          el("span", { className: "n", text: String(t.count_public) }),
        ]),
        el("p", { text: t.short }),
        el("div", { className: "tiers" }, [
          el("span", { className: "badge tier", text: "Tier " + t.tier }),
          t.stress ? el("span", { className: "badge type", text: "stress-test" }) : null,
        ]),
      ]);
      function toggle() {
        taxSelected = taxSelected === t.key ? null : t.key;
        renderTaxonomy();
        renderTaxDetail();
      }
      card.addEventListener("click", toggle);
      card.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });
      grid.appendChild(card);
    });
  }

  function renderTaxDetail() {
    var box = document.getElementById("tax-detail");
    clear(box);
    if (!taxSelected) return;
    var t = TYPE[taxSelected];
    var card = el("div", { className: "card" }, [
      el("h3", { text: t.name }),
      el("p", { className: "card-sub", text: t.description }),
      el("p", { className: "note", text: t.count_public + " entries across the public splits · difficulty tier " + t.tier + (t.stress ? " · theoretically-motivated stress-test type (zero documented real-world instances)" : "") }),
    ]);
    var ex = D.examples.filter(function (e) { return e.type === t.key; })[0];
    if (ex) card.appendChild(exampleCard(ex, true));
    box.appendChild(card);
  }

  function renderTiers() {
    var grid = document.getElementById("tier-cards");
    clear(grid);
    D.tiers.forEach(function (t) {
      grid.appendChild(el("div", { className: "card" }, [
        el("h3", {}, [
          el("span", {
            className: "swatch", style: "display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:8px;background:" + tierColor(t.tier),
          }),
          document.createTextNode(t.name),
        ]),
        el("p", { className: "card-sub", text: t.description }),
      ]));
    });
  }

  function renderSplits() {
    var grid = document.getElementById("split-cards");
    D.splits.forEach(function (s) {
      var total = s.valid + s.hallucinated;
      var card = el("div", { className: "card" }, [
        el("h3", {}, [
          document.createTextNode(s.name),
          s.extension ? el("span", { className: "cat-tag", text: "extension" }) : null,
        ]),
        el("p", { className: "card-sub", text: s.description }),
      ]);
      var stack = el("div", { className: "stackbar", role: "img", "aria-label": s.name + ": " + s.valid + " valid, " + s.hallucinated + " hallucinated" });
      stack.appendChild(el("div", {
        className: "segf",
        style: "width:" + (100 * s.valid / total) + "%;background:var(--status-good)",
      }));
      stack.appendChild(el("div", {
        className: "segf",
        style: "width:" + (100 * s.hallucinated / total) + "%;background:var(--status-critical)",
      }));
      card.appendChild(stack);
      card.appendChild(el("div", { className: "stack-labels" }, [
        el("span", {}, [el("b", { text: String(s.valid) }), " valid"]),
        el("span", {}, [el("b", { text: String(s.hallucinated) }), " hallucinated"]),
        el("span", {}, [el("b", { text: String(total) }), " total"]),
      ]));
      grid.appendChild(card);
    });
  }

  /* ---------- results explorer ---------- */

  var RS = {
    split: D.default_split,
    cats: {},
    models: {},
    types: {},
  };
  D.categories.forEach(function (c) { RS.cats[c.key] = true; });
  D.models.forEach(function (m) { RS.models[m.tool] = m.default_on; });
  D.taxonomy.forEach(function (t) { RS.types[t.key] = true; });

  var sortState = { key: "f1", dir: -1 };

  function activeResults() {
    var rows = D.results[RS.split] || [];
    return rows.filter(function (r) {
      var m = MODEL[r.tool];
      return m && RS.cats[m.category] && RS.models[r.tool];
    });
  }

  function renderResultFilters() {
    var seg = document.getElementById("split-seg");
    clear(seg);
    Object.keys(D.results).forEach(function (sk) {
      var meta = D.split_labels[sk] || sk;
      var b = el("button", {
        className: RS.split === sk ? "on" : "", text: meta, role: "tab",
      });
      b.addEventListener("click", function () {
        RS.split = sk;
        renderResultFilters();
        renderResults();
      });
      seg.appendChild(b);
    });

    var catBox = document.getElementById("cat-chips");
    clear(catBox);
    D.categories.forEach(function (c) {
      var chip = el("button", { className: "chip" + (RS.cats[c.key] ? " on" : "") }, [
        el("span", { className: "hm-dot", style: "background:" + CAT_COLOR[c.key] }),
        c.name,
      ]);
      chip.addEventListener("click", function () {
        RS.cats[c.key] = !RS.cats[c.key];
        renderResultFilters();
        renderResults();
      });
      catBox.appendChild(chip);
    });

    var mBox = document.getElementById("model-chips");
    clear(mBox);
    var present = {};
    (D.results[RS.split] || []).forEach(function (r) { present[r.tool] = true; });
    D.models.filter(function (m) { return present[m.tool] && RS.cats[m.category]; })
      .forEach(function (m) {
        var chip = el("button", { className: "chip" + (RS.models[m.tool] ? " on" : "") }, [
          el("span", { className: "hm-dot", style: "background:" + CAT_COLOR[m.category] }),
          m.name,
        ]);
        chip.addEventListener("click", function () {
          RS.models[m.tool] = !RS.models[m.tool];
          renderResultFilters();
          renderResults();
        });
        mBox.appendChild(chip);
      });
  }

  var LB_COLS = [
    { key: "name", label: "Verifier", num: false },
    { key: "dr", label: "Detection rate", num: true, fmt: function (r) { return fmt(r.dr); } },
    { key: "fpr", label: "FPR", num: true, fmt: function (r) { return fmt(r.fpr); }, lowerBetter: true },
    { key: "f1", label: "F1", num: true, fmt: function (r) { return fmt(r.f1); } },
    { key: "twf1", label: "Tier-wtd F1", num: true, fmt: function (r) { return fmt(r.twf1); } },
    { key: "tier3_f1", label: "Tier-3 F1", num: true, fmt: function (r) { return fmt(r.tier3_f1); } },
    { key: "ece", label: "ECE", num: true, fmt: function (r) { return fmt(r.ece); }, lowerBetter: true },
    { key: "auroc", label: "AUROC", num: true, fmt: function (r) { return fmt(r.auroc); } },
    { key: "coverage", label: "Coverage", num: true, fmt: function (r) { return pct(r.coverage, 0); } },
  ];

  function renderLeaderboard() {
    var box = document.getElementById("leaderboard");
    clear(box);
    var rows = activeResults().slice();
    rows.sort(function (a, b) {
      var ka = sortState.key === "name" ? modelName(a.tool) : a[sortState.key];
      var kb = sortState.key === "name" ? modelName(b.tool) : b[sortState.key];
      var kaNull = ka === null || ka === undefined;
      var kbNull = kb === null || kb === undefined;
      if (kaNull && kbNull) return 0;
      if (kaNull) return 1;
      if (kbNull) return -1;
      return (ka < kb ? -1 : ka > kb ? 1 : 0) * sortState.dir;
    });

    var best = {};
    LB_COLS.forEach(function (c) {
      if (!c.num) return;
      var vals = rows
        .filter(function (r) { return MODEL[r.tool].ranked && r[c.key] !== null && r[c.key] !== undefined; })
        .map(function (r) { return r[c.key]; });
      if (!vals.length) return;
      best[c.key] = c.lowerBetter ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
    });

    var thead = el("tr");
    LB_COLS.forEach(function (c) {
      var th = el("th", {}, [c.label,
        sortState.key === c.key ? el("span", { className: "arrow", text: sortState.dir === -1 ? " ▼" : " ▲" }) : null]);
      th.addEventListener("click", function () {
        if (sortState.key === c.key) sortState.dir *= -1;
        else { sortState.key = c.key; sortState.dir = c.num ? -1 : 1; }
        renderLeaderboard();
      });
      thead.appendChild(th);
    });

    var tbody = el("tbody");
    rows.forEach(function (r) {
      var m = MODEL[r.tool];
      var tr = el("tr");
      tr.appendChild(el("td", { className: "name-cell" }, [
        el("div", { className: "name-flex" }, [
          el("span", { className: "hm-dot", style: "background:" + CAT_COLOR[m.category] }),
          el("span", { text: m.name }),
          m.tag ? el("span", { className: "cat-tag", text: m.tag }) : null,
        ]),
      ]));
      LB_COLS.slice(1).forEach(function (c) {
        var isBest = m.ranked && best[c.key] !== undefined && r[c.key] === best[c.key];
        tr.appendChild(el("td", { className: isBest ? "best" : "", text: c.fmt(r) }));
      });
      tbody.appendChild(tr);
    });

    var table = el("table", { className: "data" }, [el("thead", {}, [thead]), tbody]);
    box.appendChild(table);
    var note = document.getElementById("leaderboard-note");
    note.textContent = (D.split_notes[RS.split] || "") +
      " Coverage = share of entries the verifier commits to (non-abstained). Bold = best among ranked verifiers.";
  }

  function renderScatter() {
    var box = document.getElementById("scatter");
    clear(box);
    var rows = activeResults().filter(function (r) { return r.dr !== null && r.fpr !== null; });
    var W = 760, H = 486, padL = 58, padR = 40, padT = 34, padB = 52;
    var xmax = Math.min(1, Math.max(0.3, Math.max.apply(null, rows.map(function (r) { return r.fpr; }).concat([0])) * 1.12));
    function x(v) { return padL + (v / xmax) * (W - padL - padR); }
    function y(v) { return padT + (1 - v) * (H - padT - padB); }
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", role: "img", "aria-label": "Detection rate versus false-positive rate scatter" });

    for (var gy = 0; gy <= 1.0001; gy += 0.25) {
      svg.appendChild(svgEl("line", { x1: padL, x2: W - padR, y1: y(gy), y2: y(gy), class: "gridline" }));
      svg.appendChild(svgEl("text", { x: padL - 8, y: y(gy) + 4, "text-anchor": "end", class: "ticklab", text: Math.round(gy * 100) + "%" }));
    }
    var xstep = xmax > 0.5 ? 0.2 : 0.1;
    for (var gx = 0; gx <= xmax + 1e-9; gx += xstep) {
      svg.appendChild(svgEl("line", { x1: x(gx), x2: x(gx), y1: padT, y2: H - padB, class: "gridline" }));
      svg.appendChild(svgEl("text", { x: x(gx), y: H - padB + 18, "text-anchor": "middle", class: "ticklab", text: Math.round(gx * 100) + "%" }));
    }
    svg.appendChild(svgEl("line", { x1: padL, x2: padL, y1: padT, y2: H - padB, class: "axisline" }));
    svg.appendChild(svgEl("line", { x1: padL, x2: W - padR, y1: H - padB, y2: H - padB, class: "axisline" }));
    svg.appendChild(svgEl("text", { x: (padL + W - padR) / 2, y: H - 8, "text-anchor": "middle", class: "axistitle", text: "False-positive rate on valid entries →" }));
    svg.appendChild(svgEl("text", { x: 14, y: (padT + H - padB) / 2, class: "axistitle", transform: "rotate(-90 14 " + (padT + H - padB) / 2 + ")", "text-anchor": "middle", text: "Detection rate on hallucinated entries →" }));

    // label placement with simple collision avoidance; dots count as occupied
    var placed = [];
    rows.forEach(function (r) {
      placed.push({ x: x(r.fpr), y: y(r.dr), w: 16 });
    });
    function labelFits(bx) {
      return !placed.some(function (p) {
        return Math.abs(p.x - bx.x) < (p.w + bx.w) / 2 + 4 && Math.abs(p.y - bx.y) < 13;
      });
    }

    rows.forEach(function (r) {
      var m = MODEL[r.tool];
      var cx = x(r.fpr), cy = y(r.dr);
      var color = CAT_COLOR[m.category];
      var g = svgEl("g", { tabindex: "0" });
      g.appendChild(svgEl("circle", { cx: cx, cy: cy, r: 6, fill: color, stroke: "var(--surface-1)", "stroke-width": 2 }));
      // direct label: try right, then left, then above; tooltip carries it if no room
      var name = m.name;
      var wEst = name.length * 6.8 + 4;
      var candidates = [
        { x: cx + 12 + wEst / 2, y: cy + 4, anchor: "start", tx: cx + 12 },
        { x: cx - 12 - wEst / 2, y: cy + 4, anchor: "end", tx: cx - 12 },
        { x: cx, y: cy - 14, anchor: "middle", tx: cx },
        { x: cx, y: cy + 20, anchor: "middle", tx: cx },
      ];
      for (var i = 0; i < candidates.length; i++) {
        var c = candidates[i];
        var bx = { x: c.x, y: c.y, w: wEst };
        var inside = c.y > 12 && c.y < H - padB - 4 &&
          c.tx - (c.anchor === "end" ? wEst : c.anchor === "middle" ? wEst / 2 : 0) > padL &&
          c.tx + (c.anchor === "start" ? wEst : c.anchor === "middle" ? wEst / 2 : 0) < W - 4;
        if (inside && labelFits(bx)) {
          placed.push(bx);
          g.appendChild(svgEl("text", { x: c.tx, y: c.y, "text-anchor": c.anchor, class: "directlab", text: name }));
          break;
        }
      }
      var hit = svgEl("circle", { cx: cx, cy: cy, r: 14, fill: "transparent" });
      addTip(hit, function () { return name; }, function () {
        return [
          { value: fmt(r.dr), label: "detection rate", color: color },
          { value: fmt(r.fpr), label: "false-positive rate", color: color },
          { value: fmt(r.f1), label: "F1", color: color },
        ];
      });
      g.appendChild(hit);
      svg.appendChild(g);
    });
    box.appendChild(svg);

    var legend = document.getElementById("scatter-legend");
    clear(legend);
    D.categories.filter(function (c) { return RS.cats[c.key]; }).forEach(function (c) {
      legend.appendChild(el("span", { className: "item" }, [
        el("span", { className: "swatch", style: "border-radius:50%;background:" + CAT_COLOR[c.key] }),
        c.name,
      ]));
    });

    // table view twin
    var tbox = document.getElementById("scatter-table");
    clear(tbox);
    var tb = el("tbody");
    rows.forEach(function (r) {
      tb.appendChild(el("tr", {}, [
        el("td", { className: "name-cell", text: modelName(r.tool) }),
        el("td", { text: fmt(r.dr) }),
        el("td", { text: fmt(r.fpr) }),
      ]));
    });
    tbox.appendChild(el("table", { className: "data" }, [
      el("thead", {}, [el("tr", {}, [
        el("th", { text: "Verifier" }), el("th", { text: "Detection rate" }), el("th", { text: "FPR" }),
      ])]),
      tb,
    ]));
  }

  function renderHeatmap() {
    var box = document.getElementById("heatmap");
    clear(box);
    var rows = activeResults().filter(function (r) { return r.per_type; })
      .sort(byCategoryThenName);
    var types = D.taxonomy.filter(function (t) { return RS.types[t.key]; });

    if (!types.length || !rows.length) {
      box.appendChild(el("p", {
        className: "note",
        text: !types.length ? "No hallucination types selected." : "No verifiers selected (or none report per-type metrics on this split).",
      }));
      clear(document.getElementById("heatmap-table"));
      clear(document.getElementById("heat-scale"));
      return;
    }

    var grid = el("div", {
      className: "heatmap",
      style: "grid-template-columns: minmax(140px,190px) repeat(" + types.length + ", minmax(34px,1fr));",
    });
    grid.appendChild(el("div", { className: "heat-corner" }));
    types.forEach(function (t) {
      grid.appendChild(el("div", { className: "heat-collab", text: t.name }));
    });
    rows.forEach(function (r) {
      grid.appendChild(el("div", { className: "heat-rowlab", text: modelName(r.tool) }));
      types.forEach(function (t) {
        var cellData = r.per_type[t.key];
        if (!cellData || cellData.dr === null || cellData.dr === undefined) {
          grid.appendChild(el("div", { className: "heat-cell empty", "aria-label": "no data" }));
          return;
        }
        var cell = el("div", {
          className: "heat-cell", tabIndex: 0,
          style: "background:" + seqColor(cellData.dr),
        });
        addTip(cell, function () { return modelName(r.tool) + " × " + t.name; }, function () {
          return [
            { value: pct(cellData.dr), label: "detection rate" },
            { value: String(cellData.count), label: "entries" },
          ];
        });
        grid.appendChild(cell);
      });
    });
    box.appendChild(grid);

    var scale = document.getElementById("heat-scale");
    clear(scale);
    var stops = isDark() ? SEQ_LIGHT.slice().reverse() : SEQ_LIGHT;
    scale.appendChild(document.createTextNode("detection rate 0%"));
    scale.appendChild(el("span", { className: "ramp", style: "background:linear-gradient(90deg," + stops.join(",") + ")" }));
    scale.appendChild(document.createTextNode("100% · hatched = no entries of that type in this split"));

    // table view twin
    var tbox = document.getElementById("heatmap-table");
    clear(tbox);
    var thead = el("tr", {}, [el("th", { text: "Verifier" })].concat(types.map(function (t) {
      return el("th", { text: t.name });
    })));
    var tb = el("tbody");
    rows.forEach(function (r) {
      tb.appendChild(el("tr", {}, [el("td", { className: "name-cell", text: modelName(r.tool) })].concat(
        types.map(function (t) {
          var c = r.per_type[t.key];
          return el("td", { text: c ? fmt(c.dr, 2) : "–" });
        })
      )));
    });
    tbox.appendChild(el("table", { className: "data" }, [el("thead", {}, [thead]), tb]));
  }

  function renderTypeChips() {
    var box = document.getElementById("type-chips");
    clear(box);
    var allOn = D.taxonomy.every(function (t) { return RS.types[t.key]; });
    var allChip = el("button", { className: "chip" + (allOn ? " on" : ""), text: "all" });
    allChip.addEventListener("click", function () {
      D.taxonomy.forEach(function (t) { RS.types[t.key] = !allOn; });
      renderTypeChips();
      renderHeatmap();
    });
    box.appendChild(allChip);
    D.taxonomy.forEach(function (t) {
      var chip = el("button", { className: "chip" + (RS.types[t.key] ? " on" : ""), text: t.name });
      chip.addEventListener("click", function () {
        RS.types[t.key] = !RS.types[t.key];
        renderTypeChips();
        renderHeatmap();
      });
      box.appendChild(chip);
    });
  }

  function renderTierBars() {
    var box = document.getElementById("tier-bars");
    clear(box);
    var rows = activeResults().filter(function (r) { return r.per_tier; })
      .sort(byCategoryThenName);
    var wrap = el("div", { className: "bar-rows", style: "gap:14px" });
    rows.forEach(function (r) {
      var group = el("div", { className: "bar-row", style: "grid-template-rows:auto" }, [
        el("span", { className: "name", text: modelName(r.tool) }),
      ]);
      var tracks = el("div", { style: "display:grid;gap:2px" });
      [1, 2, 3].forEach(function (tier) {
        var td = r.per_tier[tier];
        var v = td ? td.dr : null;
        var track = el("div", { className: "track", style: "height:12px;position:relative;border-left:1px solid var(--baseline)" });
        if (v !== null) {
          track.appendChild(el("div", {
            className: "fill",
            style: "width:" + Math.max(0.5, v * 100) + "%;background:" + tierColor(tier),
          }));
        }
        addTip(track, function () { return modelName(r.tool) + " · Tier " + tier; }, function () {
          return [{ value: v === null ? "–" : pct(v), label: "detection rate", color: tierColor(tier) }];
        });
        tracks.appendChild(track);
      });
      group.appendChild(tracks);
      var t3 = r.per_tier[3];
      group.appendChild(el("span", { className: "val", text: t3 ? pct(t3.dr, 0) : "–" }));
      wrap.appendChild(group);
    });
    box.appendChild(wrap);

    var legend = document.getElementById("tier-legend");
    clear(legend);
    [1, 2, 3].forEach(function (tier) {
      legend.appendChild(el("span", { className: "item" }, [
        el("span", { className: "swatch", style: "background:" + tierColor(tier) }),
        "Tier " + tier,
      ]));
    });
    legend.appendChild(el("span", { className: "item", style: "color:var(--text-muted)", text: "value label = Tier 3" }));

    var tbox = document.getElementById("tiers-table");
    clear(tbox);
    var tb = el("tbody");
    rows.forEach(function (r) {
      tb.appendChild(el("tr", {}, [
        el("td", { className: "name-cell", text: modelName(r.tool) }),
        el("td", { text: r.per_tier[1] ? fmt(r.per_tier[1].dr, 2) : "–" }),
        el("td", { text: r.per_tier[2] ? fmt(r.per_tier[2].dr, 2) : "–" }),
        el("td", { text: r.per_tier[3] ? fmt(r.per_tier[3].dr, 2) : "–" }),
      ]));
    });
    tbox.appendChild(el("table", { className: "data" }, [
      el("thead", {}, [el("tr", {}, [
        el("th", { text: "Verifier" }), el("th", { text: "Tier 1 DR" }),
        el("th", { text: "Tier 2 DR" }), el("th", { text: "Tier 3 DR" }),
      ])]),
      tb,
    ]));
  }

  function renderResults() {
    renderLeaderboard();
    renderScatter();
    renderHeatmap();
    renderTierBars();
  }

  document.querySelectorAll(".table-toggle").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var id = btn.getAttribute("data-table-for");
      var map = { scatter: ["scatter", "scatter-table"], heatmap: ["heatmap", "heatmap-table"], tiers: ["tier-bars", "tiers-table"] };
      var pair = map[id];
      var chart = document.getElementById(pair[0]);
      var table = document.getElementById(pair[1]);
      var showingTable = !table.classList.contains("hidden");
      table.classList.toggle("hidden", showingTable);
      chart.classList.toggle("hidden", !showingTable);
      btn.textContent = showingTable ? "View as table" : "View as chart";
    });
  });

  /* ---------- examples browser ---------- */

  var EX = { label: "all", type: "all", tier: "all", method: "all", shown: 24 };

  function bibtexOf(e) {
    var order = ["author", "title", "booktitle", "journal", "year", "volume", "pages", "doi", "url", "pmid"];
    var keys = order.filter(function (k) { return e.fields[k] !== undefined && e.fields[k] !== null; })
      .concat(Object.keys(e.fields).filter(function (k) { return order.indexOf(k) === -1; }).sort());
    var lines = keys.map(function (k) { return "  " + k + " = {" + e.fields[k] + "}"; });
    return "@" + e.bibtex_type + "{" + e.key + ",\n" + lines.join(",\n") + "\n}";
  }

  var SUBTEST_NAMES = {
    doi_resolves: "DOI resolves",
    title_exists: "title exists",
    authors_match: "authors match",
    venue_correct: "venue correct",
    fields_complete: "fields complete",
    cross_db_agreement: "cross-DB agreement",
  };

  function exampleCard(e, open) {
    var t = e.type ? TYPE[e.type] : null;
    var details = el("details", { className: "example-card" });
    if (open) details.setAttribute("open", "");
    var summary = el("summary", {}, [
      el("span", { className: "twist", text: "▶" }),
      el("span", { className: "ex-title", text: (e.fields.title || e.key) }),
      el("span", { className: "badge " + (e.label === "VALID" ? "valid" : "hallucinated"), text: e.label }),
      t ? el("span", { className: "badge type", text: t.name }) : null,
      e.tier ? el("span", { className: "badge tier", text: "Tier " + e.tier }) : null,
    ]);
    details.appendChild(summary);

    var body = el("div", { className: "example-body" });
    body.appendChild(el("pre", { className: "bibtex", text: bibtexOf(e) }));
    if (e.explanation) {
      body.appendChild(el("div", { className: "explain" }, [
        el("b", { text: e.label === "VALID" ? "Why it is valid: " : "Ground-truth diagnosis: " }),
        document.createTextNode(e.explanation),
      ]));
    }
    var subs = el("div", { className: "subtests" });
    Object.keys(SUBTEST_NAMES).forEach(function (k) {
      if (!e.subtests || e.subtests[k] === null || e.subtests[k] === undefined) return;
      subs.appendChild(el("span", {
        className: "subtest " + (e.subtests[k] ? "pass" : "fail"),
        text: (e.subtests[k] ? "✓ " : "✗ ") + SUBTEST_NAMES[k],
      }));
    });
    body.appendChild(subs);
    body.appendChild(el("p", { className: "note", style: "margin-top:10px", text: "generated by: " + (e.method || "unknown") + (e.venue ? " · source venue: " + e.venue : "") + " · key: " + e.key }));
    details.appendChild(body);
    return details;
  }

  function exFiltered() {
    return D.examples.filter(function (e) {
      if (EX.label !== "all" && e.label !== EX.label) return false;
      if (EX.type !== "all" && e.type !== EX.type) return false;
      if (EX.tier !== "all" && String(e.tier) !== EX.tier) return false;
      if (EX.method !== "all" && e.method !== EX.method) return false;
      return true;
    });
  }

  function chipRow(boxId, options, getter, setter) {
    var box = document.getElementById(boxId);
    clear(box);
    options.forEach(function (o) {
      var chip = el("button", { className: "chip" + (getter() === o.value ? " on" : ""), text: o.label });
      chip.addEventListener("click", function () {
        setter(getter() === o.value ? "all" : o.value);
        EX.shown = 24;
        renderExampleFilters();
        renderExamples();
      });
      box.appendChild(chip);
    });
  }

  function renderExampleFilters() {
    chipRow("ex-label-chips", [
      { value: "all", label: "all" },
      { value: "HALLUCINATED", label: "hallucinated" },
      { value: "VALID", label: "valid" },
    ], function () { return EX.label; }, function (v) { EX.label = v; });

    chipRow("ex-type-chips", [{ value: "all", label: "all" }].concat(D.taxonomy.map(function (t) {
      return { value: t.key, label: t.name };
    })), function () { return EX.type; }, function (v) { EX.type = v; });

    chipRow("ex-tier-chips", [
      { value: "all", label: "all" },
      { value: "1", label: "Tier 1" }, { value: "2", label: "Tier 2" }, { value: "3", label: "Tier 3" },
    ], function () { return EX.tier; }, function (v) { EX.tier = v; });

    chipRow("ex-method-chips", [{ value: "all", label: "all" }].concat(D.generation_methods.map(function (m) {
      return { value: m.key, label: m.name };
    })), function () { return EX.method; }, function (v) { EX.method = v; });
  }

  function renderExamples() {
    var list = document.getElementById("examples-list");
    clear(list);
    var rows = exFiltered();
    rows.slice(0, EX.shown).forEach(function (e) { list.appendChild(exampleCard(e)); });
    document.getElementById("examples-count").textContent =
      rows.length + " of " + D.examples.length + " sampled entries match (corpus: " + D.corpus_note + ")";
    var more = document.getElementById("examples-more");
    more.style.display = rows.length > EX.shown ? "" : "none";
  }

  document.getElementById("examples-more").addEventListener("click", function () {
    EX.shown += 24;
    renderExamples();
  });

  /* ---------- cite ---------- */

  document.getElementById("copy-bibtex").addEventListener("click", function () {
    var text = document.getElementById("bibtex-block").textContent;
    var btn = this;
    function done() {
      btn.textContent = "Copied";
      setTimeout(function () { btn.textContent = "Copy"; }, 1600);
    }
    function fallback() {
      var ta = el("textarea", { value: text, style: "position:fixed;opacity:0" });
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      done();
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }
  });

  /* ---------- boot ---------- */

  function rerenderThemed() {
    renderFailureModes();
    renderTiers();
    renderResults();
  }

  document.querySelectorAll(".corpus-version").forEach(function (n) {
    n.textContent = "v" + D.corpus_version;
  });
  document.getElementById("footer-generated").textContent =
    "Data regenerated from the released result files on " + D.generated + ".";

  renderFailureModes();
  renderTaxonomy();
  renderTiers();
  renderSplits();
  renderResultFilters();
  renderTypeChips();
  renderResults();
  renderExampleFilters();
  renderExamples();
})();
