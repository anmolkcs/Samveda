/* Samadhvani demo client.  No framework: the whole UI is one screen and the
   interesting state is the last render, so plain DOM keeps it legible. */

const $ = (id) => document.getElementById(id);
const SVARA_LABEL = { 1: 'kṛṣṭa', 2: 'prathama', 3: 'dvitīya', 4: 'tṛtīya',
                      5: 'caturtha', 6: 'mandra', 7: 'atisvārya' };
const SVARA_DEVA  = { 1: 'क्रुष्ट', 2: 'प्रथम', 3: 'द्वितीय', 4: 'तृतीय',
                      5: 'चतुर्थ', 6: 'मन्द्र', 7: 'अतिस्वार्य' };

let SAMPLES = [];
let health = {};

/* ── tabs ─────────────────────────────────────────────────── */
document.querySelectorAll('nav button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'data') loadData();
  };
});

/* ── slider outputs ───────────────────────────────────────── */
const bind = (id, out, fmt) => {
  const el = $(id), o = $(out);
  const upd = () => o.textContent = fmt(el.value);
  el.addEventListener('input', upd); upd();
};
bind('laya', 'layaOut', v => (+v).toFixed(2) + ' s');
bind('tonic', 'tonicOut', v => v + ' Hz');
bind('transpose', 'trOut', v => (v > 0 ? '+' : '') + v + ' st');
bind('intensity', 'intOut', v => (+v).toFixed(2));
bind('room', 'roomOut', v => (+v).toFixed(2));

/* ── boot ─────────────────────────────────────────────────── */
async function boot() {
  try {
    health = await (await fetch('/api/health')).json();
  } catch { health = {}; }

  const eng = $('engine');
  eng.innerHTML = '';
  const engines = (health.engines && health.engines.length) ? health.engines : ['formant'];
  const LABELS = {
    formant: 'formant (built in)',
    voice: 'voice (trained on sāmagāna)',
    neural: 'neural (trained checkpoint)',
  };
  for (const e of engines) {
    const o = document.createElement('option');
    o.value = e;
    o.textContent = LABELS[e] || e;
    eng.appendChild(o);
  }
  for (const e of ['voice', 'neural']) {
    if (engines.includes(e)) continue;
    const o = document.createElement('option');
    o.value = e; o.disabled = true;
    o.textContent = `${e} — train a checkpoint first`;
    eng.appendChild(o);
  }
  if (engines.includes('voice')) eng.value = 'voice';

  try {
    const meta = await (await fetch('/api/samples')).json();
    SAMPLES = meta.samples || [];
    const sel = $('samples');
    SAMPLES.forEach((s, i) => {
      const o = document.createElement('option');
      o.value = i; o.textContent = s.title;
      sel.appendChild(o);
    });
    sel.onchange = () => {
      const s = SAMPLES[+sel.value];
      if (!s) return;
      $('text').value = s.sanskrit;
      if (s.mode_hint) $('mode').value = s.mode_hint;
      $('status').textContent = s.note;
      $('status').className = 'status';
    };

    const sch = $('school');
    (meta.schools || []).forEach(s => {
      const o = document.createElement('option');
      o.value = s.key;
      o.textContent = `${s.name} · ${s.iast}`;
      o.title = s.notes;
      sch.appendChild(o);
    });
    sch.onchange = () => {
      const s = (meta.schools || []).find(x => x.key === sch.value);
      if (s) { $('status').textContent = s.notes; $('status').className = 'status'; }
    };

    const mdl = $('model');
    (meta.models || []).forEach(m => {
      const o = document.createElement('option');
      o.value = m.key;
      o.textContent = `${m.name_deva} · ${m.name_iast}`;
      o.title = m.notes;
      mdl.appendChild(o);
    });

    // default text
    $('text').value = (SAMPLES.find(s => s.id === 'agnimile') || SAMPLES[0] || {}).sanskrit || '';
  } catch (e) {
    $('status').textContent = 'Could not load options: ' + e;
    $('status').className = 'status err';
  }
}

/* ── render ───────────────────────────────────────────────── */
async function sing() {
  const text = $('text').value.trim();
  if (!text) { return; }
  const btn = $('sing');
  btn.disabled = true;
  $('status').textContent = 'chanting…';
  $('status').className = 'status';

  const body = {
    text,
    mode: $('mode').value,
    school: $('school').value || 'kauthuma',
    model: $('model').value || 'gramageya',
    laya: +$('laya').value,
    tonic_hz: +$('tonic').value,
    transpose: +$('transpose').value,
    intensity: +$('intensity').value,
    room: +$('room').value,
    engine: $('engine').value,
    evaluate: $('evaluate').checked,
  };

  try {
    const res = await fetch('/api/chant', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    present(data);
    $('status').textContent =
      `rendered ${data.notes.length} notes in ${data.render_ms.toFixed(0)} ms · engine ${body.engine}`;
    $('status').className = 'status ok';
  } catch (e) {
    $('status').textContent = 'Error: ' + e.message;
    $('status').className = 'status err';
  } finally {
    btn.disabled = false;
  }
}

function present(d) {
  $('placeholder').classList.add('hidden');
  $('playerWrap').classList.remove('hidden');

  const badge = $('modeBadge');
  badge.textContent = d.mode;
  badge.className = 'badge ' + d.mode;

  const audio = $('player');
  audio.src = 'data:' + d.audio_mime + ';base64,' + d.audio_b64;
  audio.play().catch(() => {});

  const meta = [
    ['duration', d.duration.toFixed(2) + ' s'],
    ['syllables', d.syllables.length],
    ['chanda', d.chanda || '—'],
    ['śākhā', d.school],
    ['model', d.model],
    ['tonic', d.tonic_hz + ' Hz'],
    ['script', d.script_detected],
  ];
  $('meta').innerHTML = meta
    .map(([k, v]) => `<span><b>${k}</b> ${v}</span>`).join('');

  drawStrip(d);
  drawSyllables(d);
  drawAnalysis(d);
  drawCaveats(d);
}

/* ── svara strip: one bar per note, height = svara pitch ──── */
function drawStrip(d) {
  const wrap = $('svaraStrip');
  wrap.classList.remove('hidden');
  const notes = d.notes || [];
  if (!notes.length) { wrap.innerHTML = ''; return; }

  const maxDur = Math.max(...notes.map(n => n.duration));
  let html = `<h3 style="margin-top:0">Svara track <span class="small">— bar height is the note,
    width is how long it is held; the akṣara is under it</span></h3><div>`;
  for (const n of notes) {
    const s = n.svara;
    const h = 12 + (7 - s) * 7;                       // higher svara number = lower note
    const cls = s <= 3 ? 'high' : (s >= 6 ? 'low' : '');
    const w = Math.round(26 + 26 * (n.duration / maxDur));
    html += `<span class="note" title="${SVARA_DEVA[s]} (${SVARA_LABEL[s]}) · ${n.f0_hz} Hz · ${n.duration}s">
        <span class="nam">${s}</span>
        <span class="bar ${cls}" style="height:${h}px;width:${w}px"></span>
        <span class="glyph">${n.text || '·'}</span>
      </span>`;
  }
  wrap.innerHTML = html + '</div>';
}

/* ── per-syllable table ───────────────────────────────────── */
function drawSyllables(d) {
  $('breakdown').classList.remove('hidden');
  const notes = d.notes || [];
  let html = `<table><thead><tr>
      <th>#</th><th>akṣara</th><th>phones</th><th>svara</th><th>Hz</th><th>t</th><th>dur</th>
    </tr></thead><tbody>`;
  notes.forEach((n, i) => {
    html += `<tr>
      <td class="mono">${i + 1}</td>
      <td class="deva">${n.text || '·'}</td>
      <td class="mono">${(n.phones || []).join(' ')}</td>
      <td><span class="pill on">${n.svara} ${SVARA_LABEL[n.svara] || ''}</span></td>
      <td class="mono">${n.f0_hz}</td>
      <td class="mono">${n.start.toFixed(2)}</td>
      <td class="mono">${n.duration.toFixed(2)}</td>
    </tr>`;
  });
  $('syllables').innerHTML = html + '</tbody></table>';
  $('syllables').classList.add('scroll');
}

/* ── vikaras + metrics ────────────────────────────────────── */
function drawAnalysis(d) {
  let html = '<h4 style="margin:4px 0 6px;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#7a6d5e">vikāras applied</h4>';
  const names = {
    vikara: 'vikāra', vislesa: 'viśleṣa', vikarshana: 'vikarṣaṇa',
    abhyasa: 'abhyāsa', virama: 'virāma', stobha: 'stobha', agama: 'āgama',
  };
  for (const [k, v] of Object.entries(d.vikaras || {})) {
    html += `<span class="pill ${v ? 'on' : 'off'}" style="margin:2px 4px 2px 0">${names[k] || k}</span>`;
  }

  const ev = d.evaluation;
  if (ev && !ev.error) {
    const s = ev.svara || {}, a = ev.audio || {};
    html += '<h4 style="margin:14px 0 6px;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#7a6d5e">melody</h4>';
    if (s.svara_accuracy !== undefined) {
      html += metric('svara accuracy', (100 * s.svara_accuracy).toFixed(0) + ' %');
    }
    if (s.median_abs_cents_error !== undefined) {
      html += metric('median pitch error', s.median_abs_cents_error.toFixed(1) + ' cents');
    }
    html += '<h4 style="margin:14px 0 6px;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#7a6d5e">signal</h4>';
    if (a.periodicity !== undefined) html += metric('periodicity', a.periodicity.toFixed(3));
    if (a.hnr_db !== undefined) html += metric('HNR', a.hnr_db.toFixed(1) + ' dB');
    if (a.spectral_flatness !== undefined) html += metric('spectral flatness', a.spectral_flatness.toFixed(3));
    if (a.mos_proxy !== undefined) html += metric('mos proxy <span class="small">(not a MOS)</span>', a.mos_proxy.toFixed(2));
  } else if (ev && ev.error) {
    html += `<p class="small">evaluation failed: ${ev.error}</p>`;
  }
  $('analysis').innerHTML = html;
}

const metric = (k, v) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`;

function drawCaveats(d) {
  const w = $('caveats');
  const list = d.caveats || [];
  if (!list.length) { w.classList.add('hidden'); return; }
  w.classList.remove('hidden');
  w.innerHTML = list.map(c => `<div class="c">${c}</div>`).join('');
}

/* ── data audit tab ───────────────────────────────────────── */
let dataLoaded = false;
async function loadData() {
  if (dataLoaded) return;
  dataLoaded = true;
  try {
    const d = await (await fetch('/api/data')).json();
    const r = d.report || {};
    $('dataSummary').innerHTML =
      `<span class="big">${(r.audio_hours_trainable || 0).toFixed(1)} h</span> of sāmagāna trainable
       under the <b>${r.profile}</b> profile
       (${(r.audio_hours_aligned || 0).toFixed(1)} h with line-level timings) ·
       <span class="big">${(r.audio_hours_releasable || 0).toFixed(1)} h</span> openly licensed ·
       <span class="big">${r.total_ganas_in_text || 0}</span> gāna in text<br>
       <span class="small">Two independent questions, kept apart: <b>may train</b> is what this
       profile allows, <b>may ship</b> is what a licence allows. Nothing here may be
       redistributed. The profile is a promise about your conduct, not a licence.</span>`;
    let html = `<table><thead><tr>
      <th>source</th><th>kind</th><th>licence</th><th>hrs</th><th>aligned</th><th>text</th>
      <th>may train</th><th>may ship</th><th>note</th>
    </tr></thead><tbody>`;
    for (const s of d.sources) {
      const cls = s.licence === 'open' ? 'yes'
        : (s.licence === 'unclear' ? 'no' : (s.licence === 'permission' ? 'warn' : 'off'));
      const text = s.ganas ? `${s.ganas} gāna`
        : (s.verses ? `${s.verses} ṛc` : '—');
      const mirror = s.duplicate_of ? '<span class="pill warn">mirror</span> ' : '';
      html += `<tr>
        <td>${s.url ? `<a href="${s.url}" target="_blank" rel="noopener">${s.name}</a>` : s.name}</td>
        <td class="mono">${s.kind}</td>
        <td><span class="pill ${cls}">${s.licence}</span></td>
        <td class="mono">${s.hours ? s.hours.toFixed(1) : '—'}</td>
        <td class="mono">${s.aligned_hours ? s.aligned_hours.toFixed(1) : '—'}</td>
        <td class="mono">${mirror}${text}${s.svara_notation ? ' ᭄' : ''}</td>
        <td title="${s.permit_reason || ''}">${s.may_train ? '<span class="pill yes">yes</span>' : '<span class="pill no">no</span>'}</td>
        <td>${s.may_ship_weights ? '<span class="pill yes">yes</span>' : '<span class="pill no">no</span>'}</td>
        <td class="small">${s.notes}</td>
      </tr>`;
    }
    $('dataTable').innerHTML = html + '</tbody></table>';
    $('dataTable').classList.add('scroll');
  } catch (e) {
    $('dataTable').textContent = 'Could not load the data audit: ' + e;
  }
}

/* ── wire ─────────────────────────────────────────────────── */
$('sing').onclick = sing;
$('text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) sing();
});
boot();
