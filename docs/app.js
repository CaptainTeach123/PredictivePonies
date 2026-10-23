// Dashboard: reads docs/data/*.json written by `python -m ponies.pipeline`.
// No build step and no dependencies, so it runs as-is on GitHub Pages.
(async function () {
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (p, d = 1) => (p == null ? "–" : (100 * p).toFixed(d) + "%");
  const signed = (x, d = 2) => (x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(d));

  let preds, evaluation, model;
  try {
    [preds, evaluation, model] = await Promise.all(
      ["predictions", "evaluation", "model"].map((n) =>
        fetch(`data/${n}.json`).then((r) => { if (!r.ok) throw new Error(n); return r.json(); })));
  } catch (e) {
    const el = $("#load-error");
    el.hidden = false;
    el.textContent = "Could not load dashboard data (" + e.message + ".json). Run `python -m ponies.pipeline` first.";
    return;
  }
  if (preds.synthetic_data) $("#synthetic-banner").hidden = false;
  $("#generated-at").textContent = "Data generated " + preds.generated_at;
  const desc = model.feature_descriptions;

  // ---------- tabs ----------
  function selectTab(name) {
    document.querySelectorAll(".tab").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
  }
  document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
    selectTab(btn.dataset.tab);
    history.replaceState(null, "", "#" + btn.dataset.tab);
  }));
  const fromHash = location.hash.slice(1);
  if (["races", "model", "about"].includes(fromHash)) selectTab(fromHash);

  // ---------- races ----------
  const kindSel = $("#kind-filter"), raceSel = $("#race-select");
  if (!preds.races.some((r) => r.kind === "upcoming")) kindSel.value = "backtest";

  function fillRaces() {
    const races = preds.races.filter((r) => r.kind === kindSel.value);
    raceSel.innerHTML = races.map((r) =>
      `<option value="${esc(r.race_id)}">${esc(r.date)} · ${esc(r.track)} R${r.race_number}</option>`).join("");
    showRace();
  }

  function showRace() {
    const race = preds.races.find((r) => r.race_id === raceSel.value);
    const tbody = $("#race-table tbody");
    if (!race) { tbody.innerHTML = ""; $("#race-header").innerHTML = "<p>No races.</p>"; return; }
    const done = race.kind === "backtest";
    document.querySelectorAll(".result-col").forEach((c) => (c.hidden = !done));
    $("#race-header").innerHTML = `<h2>${esc(race.track)} · Race ${race.race_number} · ${esc(race.date)}</h2>
      <p>${race.distance_furlongs}f ${esc(race.surface)} (${esc(race.condition)}) · ${esc(race.race_class)}
      · purse $${(race.purse ?? 0).toLocaleString()} · ${race.field_size} runners</p>`;
    const maxP = Math.max(...race.runners.map((r) => r.model_prob));
    tbody.innerHTML = race.runners.map((r, i) => `
      <tr tabindex="0" data-i="${i}" aria-selected="false" class="${r.finish_position === 1 ? "winner" : ""}">
        <td>${r.rank}</td>
        <td>${esc(r.program)}</td>
        <td>${esc(r.horse)}${r.value_flag ? ' <span title="Value: edge above threshold" aria-label="value bet">★</span>' : ""}</td>
        <td class="hide-sm">${esc(r.jockey)}<span class="small">${esc(r.trainer)}</span></td>
        <td><div class="bar"><span class="fill" style="width:${(80 * r.model_prob) / maxP}px"></span>${pct(r.model_prob)}</div></td>
        <td class="num">${esc(r.fair_odds)}</td>
        <td class="num">${esc(r.morning_line ?? "–")}</td>
        <td class="num hide-sm">${pct(r.ml_implied_prob)}</td>
        <td class="num ${r.edge_vs_ml > 0 ? "pos" : "neg"}">${r.edge_vs_ml == null ? "–" : signed(100 * r.edge_vs_ml, 0) + "%"}</td>
        <td class="num result-col" ${done ? "" : "hidden"}>${r.finish_position ?? "–"}</td>
      </tr>`).join("");
    tbody.querySelectorAll("tr").forEach((tr) => {
      const pick = () => explain(race, +tr.dataset.i, tr);
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    });
    explain(race, 0, tbody.querySelector("tr"));
  }

  function explain(race, i, tr) {
    document.querySelectorAll("#race-table tbody tr").forEach((r) => r.setAttribute("aria-selected", r === tr));
    const r = race.runners[i];
    const items = Object.entries(r.contributions).filter(([, v]) => v != null)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8);
    const maxAbs = Math.max(0.01, ...items.map(([, v]) => Math.abs(v)));
    const rows = items.map(([f, v]) => {
      const w = (50 * Math.abs(v)) / maxAbs;
      const left = v >= 0 ? 50 : 50 - w;
      const val = r.features[f];
      return `<div title="${esc(desc[f])}">${esc(f)}<span class="small">value: ${val == null ? "missing → imputed" : val}</span></div>
        <div class="track" role="img" aria-label="${esc(f)} contribution ${signed(v)}">
          <span class="seg ${v >= 0 ? "pos" : "neg"}" style="left:${left}%;width:${w}%"></span></div>
        <div class="${v >= 0 ? "pos" : "neg"}">${signed(v)}</div>`;
    }).join("");
    $("#explain").innerHTML = `<h2>Why the model rates ${esc(r.horse)} at ${pct(r.model_prob)}</h2>
      <p class="hint">Contributions to the horse's score compared with the average rival in this race (log-odds units).
      Green pushes the win chance up, red pushes it down. Hover a feature name for its definition.</p>
      <div class="contrib">${rows}</div>
      <p class="hint">Market check: morning line ${esc(r.morning_line ?? "–")} implies ${pct(r.ml_implied_prob)}${r.final_odds ? `; final odds ${esc(r.final_odds)} implied ${pct(r.final_implied_prob)}` : ""}.</p>`;
  }

  kindSel.addEventListener("change", fillRaces);
  raceSel.addEventListener("change", showRace);
  fillRaces();

  // ---------- model tab ----------
  const ev = evaluation, ll = ev.log_loss, bt = ev.betting_backtest;
  const roi = (b) => (b.roi == null ? "–" : `${signed(100 * b.roi, 1)}% (${b.wins}/${b.bets} won)`);
  $("#metrics").innerHTML = `<dl class="metrics">
    <dt>Test races (unseen)</dt><dd>${ev.races} · ${esc(ev.date_range.join(" → "))}</dd>
    <dt>Training races</dt><dd>${ev.train.races}</dd>
    <dt>Log loss — model</dt><dd>${ll.model}</dd>
    <dt>Log loss — market (final odds)</dt><dd>${ll.market ?? "–"}</dd>
    <dt>Log loss — random guess</dt><dd>${ll.uniform}</dd>
    <dt>McFadden R² vs random</dt><dd>${ev.mcfadden_r2_vs_uniform}</dd>
    <dt>Model top pick won</dt><dd>${pct(ev.top_pick_win_rate.model)}</dd>
    <dt>Market favourite won</dt><dd>${pct(ev.top_pick_win_rate.market_favourite)}</dd>
    <dt>ROI betting model top pick</dt><dd>${roi(bt.model_top_pick)}</dd>
    <dt>ROI value bets (edge &gt; ${pct(bt.value_bets.edge_threshold, 0)})</dt><dd>${roi(bt.value_bets)}</dd>
  </dl><p class="hint">Lower log loss is better. Beating the market's log loss is the real bar.</p>`;

  // calibration scatter (inline SVG)
  const S = 260, pad = 40, sx = (p) => pad + p * (S - 2 * pad), sy = (p) => S - pad - p * (S - 2 * pad);
  const cal = ev.calibration;
  const maxN = Math.max(...cal.map((c) => c.count));
  $("#calibration").innerHTML = `<svg viewBox="0 0 ${S} ${S}" width="100%" style="max-width:${S + 60}px" role="img"
      aria-label="Calibration: predicted versus observed win rate">
    <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="currentColor" opacity=".3" stroke-dasharray="4 4"/>
    <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(0)}" stroke="currentColor" opacity=".5"/>
    <line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(0)}" y2="${sy(1)}" stroke="currentColor" opacity=".5"/>
    ${[0, 0.5, 1].map((t) => `<text x="${sx(t)}" y="${S - 12}" text-anchor="middle">${t * 100}%</text>
      <text x="${pad - 6}" y="${sy(t) + 4}" text-anchor="end">${t * 100}%</text>`).join("")}
    <text x="${S / 2}" y="${S}" text-anchor="middle">predicted</text>
    ${cal.map((c) => `<circle cx="${sx(c.mean_predicted)}" cy="${sy(c.observed_win_rate)}"
      r="${3 + 6 * Math.sqrt(c.count / maxN)}" fill="var(--accent)" opacity=".8">
      <title>${pct(c.bin_low, 0)}–${pct(c.bin_high, 0)}: predicted ${pct(c.mean_predicted)}, won ${pct(c.observed_win_rate)} (n=${c.count})</title></circle>`).join("")}
  </svg>`;

  // weights
  const ws = model.feature_names.map((f, i) => [f, model.weights[i]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxW = Math.max(...ws.map(([, w]) => Math.abs(w)));
  $("#weights").innerHTML = `<div class="contrib">${ws.map(([f, w]) => {
    const width = (50 * Math.abs(w)) / maxW, left = w >= 0 ? 50 : 50 - width;
    return `<div>${esc(f)}<span class="small">${esc(desc[f])}</span></div>
      <div class="track"><span class="seg ${w >= 0 ? "pos" : "neg"}" style="left:${left}%;width:${width}%"></span></div>
      <div class="${w >= 0 ? "pos" : "neg"}">${signed(w, 3)}</div>`;
  }).join("")}</div>`;
})();
