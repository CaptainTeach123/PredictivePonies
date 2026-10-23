// "Enter a race" tab: builds an entries XML card (same layout as DATA_SCHEMA.md)
// from the form or an uploaded file, then runs ponies/browser.py in Pyodide.
// The model code is the repo's Python package (docs/data/ponies.zip), so
// browser and offline predictions are identical.
(function () {
  const PYODIDE_VERSION = "0.27.8"; // Python 3.12, numpy 2.0, pandas 2.2
  const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
  const $ = (s, el = document) => el.querySelector(s);
  const form = $("#race-form"), tbody = $("#runner-table tbody"), status = $("#engine-status");
  const xmlEsc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
  const setStatus = (msg, isError = false) => {
    status.textContent = msg;
    status.classList.toggle("error-text", isError);
  };

  // ---------- roster (autocomplete + name -> id) ----------
  let roster = { horses: [], jockeys: [], trainers: [] };
  const byName = { horses: new Map(), jockeys: new Map(), trainers: new Map() };
  fetch("data/roster.json").then((r) => r.json()).then((data) => {
    roster = data;
    for (const [kind, list] of [["horses", "horse-list"], ["jockeys", "jockey-list"], ["trainers", "trainer-list"]]) {
      for (const p of roster[kind]) byName[kind].set(p.name.toLowerCase(), p);
      $("#" + list).innerHTML = roster[kind].map((p) => `<option value="${xmlEsc(p.name)}"></option>`).join("");
    }
    $("#roster-count").textContent = `${roster.horses.length} horses, ${roster.jockeys.length} jockeys and ${roster.trainers.length} trainers`;
  }).catch(() => {});

  // Unknown names get a stable "NEW-" id, so they're treated as having no history.
  const idFor = (kind, name) => {
    const hit = byName[kind].get(name.trim().toLowerCase());
    return hit ? hit.id : "NEW-" + name.trim().toUpperCase().replace(/[^A-Z0-9]+/g, "-");
  };

  // ---------- runner rows ----------
  const cols = [
    ["horse", "text", "horse-list", "Horse name"], ["jockey", "text", "jockey-list", "Jockey"],
    ["trainer", "text", "trainer-list", "Trainer"], ["ml", "text", null, "Morning-line odds"],
    ["weight", "number", null, "Weight (lbs)"], ["age", "number", null, "Age"], ["sex", "text", null, "Sex"],
  ];
  function addRow(values = {}) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="prog"></td>` + cols.map(([name, type, list, label]) =>
      `<td><input name="${name}" type="${type}" ${list ? `list="${list}" autocomplete="off"` : ""}
        aria-label="${label}" ${name === "weight" ? 'min="90" max="140"' : ""}
        ${name === "age" ? 'min="2" max="15"' : ""} ${name === "sex" ? 'maxlength="1" placeholder="G"' : ""}
        ${name === "ml" ? 'placeholder="5-2"' : ""} value="${xmlEsc(values[name] ?? "")}"></td>`).join("") +
      `<td><button type="button" class="remove" aria-label="Remove runner">✕</button></td>`;
    $(".remove", tr).addEventListener("click", () => { tr.remove(); renumber(); });
    $('input[name="horse"]', tr).addEventListener("change", (e) => {
      const h = byName.horses.get(e.target.value.trim().toLowerCase());
      if (!h) return;
      const fill = (n, v) => { const el = $(`input[name="${n}"]`, tr); if (!el.value && v != null) el.value = v; };
      fill("trainer", h.trainer_name); fill("age", h.age); fill("sex", h.sex);
    });
    tbody.appendChild(tr);
    renumber();
  }
  function renumber() {
    [...tbody.rows].forEach((tr, i) => ($(".prog", tr).textContent = i + 1));
  }
  for (let i = 0; i < 6; i++) addRow();
  form.elements.date.value = new Date().toISOString().slice(0, 10);
  $("#add-runner").addEventListener("click", () => addRow());

  // ---------- form -> entries XML ----------
  function formToXml() {
    const f = form.elements;
    const problems = [];
    for (const n of ["track", "date", "number", "distance"]) {
      const ok = String(f[n].value).trim() !== "";
      f[n].setAttribute("aria-invalid", !ok);
      if (!ok) problems.push(f[n].closest("label").firstChild.textContent.trim());
    }
    const runners = [...tbody.rows].map((tr) => Object.fromEntries(
      [...tr.querySelectorAll("input")].map((i) => [i.name, i.value.trim()]))).filter((r) => r.horse);
    if (runners.length < 2) problems.push("at least two runners with a horse name");
    if (problems.length) throw new Error("Please fill in: " + problems.join(", ") + ".");

    const track = f.track.value.trim().toUpperCase();
    const starters = runners.map((r, i) => {
      const a = {
        program: i + 1, post_position: i + 1,
        horse_id: idFor("horses", r.horse), horse_name: r.horse,
        jockey_id: r.jockey ? idFor("jockeys", r.jockey) : "", jockey_name: r.jockey,
        trainer_id: r.trainer ? idFor("trainers", r.trainer) : "", trainer_name: r.trainer,
        morning_line: r.ml, weight_lbs: r.weight, age: r.age, sex: r.sex.toUpperCase(),
      };
      return "    <starter " + Object.entries(a).filter(([, v]) => v !== "" && v != null)
        .map(([k, v]) => `${k}="${xmlEsc(v)}"`).join(" ") + "/>";
    });
    const purse = f.purse.value ? ` purse="${xmlEsc(f.purse.value)}"` : "";
    const cls = f.class.value.trim() ? ` class="${xmlEsc(f.class.value.trim())}"` : "";
    return `<?xml version="1.0" encoding="utf-8"?>
<entries track="${xmlEsc(track)}" date="${xmlEsc(f.date.value)}">
  <race number="${xmlEsc(f.number.value)}" distance_furlongs="${xmlEsc(f.distance.value)}" surface="${xmlEsc(f.surface.value)}" condition="${xmlEsc(f.condition.value)}"${cls}${purse}>
${starters.join("\n")}
  </race>
</entries>
`;
  }

  // ---------- Pyodide engine ----------
  let enginePromise = null;
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.onload = resolve; s.onerror = () => reject(new Error("Could not load " + src));
      document.head.appendChild(s);
    });
  }
  function engine() {
    enginePromise ??= (async () => {
      setStatus("Loading Python engine… (first time only, about 15–30 s)");
      await loadScript(PYODIDE_URL + "pyodide.js");
      const py = await loadPyodide({ indexURL: PYODIDE_URL });
      setStatus("Loading numpy and pandas…");
      await py.loadPackage(["numpy", "pandas"]);
      setStatus("Loading model and race history…");
      const [zip, historyCsv, modelJson, figureJson] = await Promise.all([
        fetch("data/ponies.zip").then((r) => { if (!r.ok) throw new Error("data/ponies.zip missing — run the pipeline"); return r.arrayBuffer(); }),
        fetch("data/history.csv").then((r) => r.text()),
        fetch("data/model.json").then((r) => r.text()),
        fetch("data/figure_model.json").then((r) => (r.ok ? r.text() : null)),
      ]);
      py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide/app" });
      py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
      const browser = py.pyimport("ponies.browser");
      browser.load_state(historyCsv, modelJson, figureJson);
      return browser;
    })();
    enginePromise.catch(() => { enginePromise = null; });
    return enginePromise;
  }

  // run(browser) returns the JSON string of race cards.
  async function predict(run) {
    const btn = $('button[type="submit"]', form);
    btn.disabled = true;
    try {
      const browser = await engine();
      setStatus("Predicting… (several files can take up to a minute)");
      await new Promise((r) => setTimeout(r, 50)); // let the status paint first
      const out = JSON.parse(await run(browser));
      const cards = Array.isArray(out) ? out : out.races;
      const skipped = Array.isArray(out) ? [] : out.skipped || [];
      const why = [...new Set(skipped.map((s) => s.reason))].join("; ");
      setStatus(`Done: ${cards.length} race${cards.length === 1 ? "" : "s"} predicted` +
        (skipped.length ? `; ${skipped.length} skipped (${why}).` : "."));
      if (cards.length) window.ponies.showCustomRaces(cards);
    } catch (e) {
      // Python exceptions arrive with a long traceback; the last line is the useful bit.
      const msg = String(e.message || e).trim().split("\n").pop();
      setStatus("Could not predict: " + msg, true);
    } finally {
      btn.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    let xml;
    try { xml = formToXml(); } catch (err) { setStatus(err.message, true); return; }
    predict((browser) => browser.predict_card(xml));
  });

  // One or more .xml / .zip files. TrackMaster files are pooled, so the
  // model trains on every file's past performances together.
  $("#xml-file").addEventListener("change", async (e) => {
    const files = [...e.target.files];
    e.target.value = "";
    if (!files.length) return;
    predict(async (browser) => {
      browser.stage_clear();
      for (const [i, f] of files.entries()) {
        setStatus(`Reading ${f.name} (${i + 1} of ${files.length})…`);
        if (/\.zip$/i.test(f.name)) browser.stage_zip(new Uint8Array(await f.arrayBuffer()));
        else browser.stage_xml(await f.text());
      }
      setStatus("Training the speed-figure model and predicting… (up to a minute for many tracks)");
      await new Promise((r) => setTimeout(r, 50));
      return browser.predict_staged();
    });
  });

  $("#download-xml").addEventListener("click", () => {
    let xml;
    try { xml = formToXml(); } catch (err) { setStatus(err.message, true); return; }
    const f = form.elements;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([xml], { type: "application/xml" }));
    a.download = `${f.track.value.trim().toUpperCase()}_${f.date.value}_R${f.number.value}.xml`;
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus("Saved. Put it in data/raw/entries/ to include it in the published dashboard.");
  });

  // Start downloading the engine as soon as someone opens the tab.
  document.querySelector('.tab[data-tab="enter"]').addEventListener("click", () =>
    engine().catch((e) => setStatus("Could not load the Python engine: " + e.message, true)), { once: true });
})();
