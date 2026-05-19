import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';
import { loadTokenizer } from './tokenizer.js';

const BLOCK_SIZE = 512;
const AMBER = [200, 146, 42];
const TEAL  = [42, 160, 180];

// ── State ─────────────────────────────────────────────────────────────────────

let session    = null;
let tokenizer  = null;
let traceData  = null;
let activeLayer = 0;
let activePos  = -1;

// ── Model ─────────────────────────────────────────────────────────────────────

async function loadModels() {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21/dist/';
  [session, tokenizer] = await Promise.all([
    ort.InferenceSession.create('./mica_trace.onnx', { executionProviders: ['wasm'] }),
    loadTokenizer(),
  ]);
}

async function runTrace(prompt) {
  const ids = tokenizer.encode(prompt).slice(-BLOCK_SIZE);
  const T   = ids.length;
  if (T === 0) throw new Error('Prompt encoded to zero tokens.');

  const input  = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, T]);
  const result = await session.run({ input_ids: input });

  const nLayer    = Object.keys(result).filter(k => k.startsWith('residual_')).length;
  const nHead     = Number(result['attn_0'].dims[1]);
  const nEmbd     = Number(result['embedding'].dims[2]);
  const vocabSize = Number(result['logits'].dims[2]);

  return {
    tokenIds:    ids,
    tokenStrings: ids.map(id => tokenizer.decode([id])),
    T, nLayer, nHead, nEmbd, vocabSize,
    embedding:   result['embedding'].data,
    residuals:   Array.from({ length: nLayer }, (_, i) => result[`residual_${i}`].data),
    attnWeights: Array.from({ length: nLayer }, (_, i) => result[`attn_${i}`].data),
    logits:      result['logits'].data,
    final:       result['final'].data,
    logitLens:   Array.from({ length: nLayer }, (_, i) => result[`logit_lens_${i}`]?.data ?? null),
  };
}

// ── Drawing ───────────────────────────────────────────────────────────────────

function cs(T) {
  return Math.max(6, Math.min(40, Math.floor(380 / T)));
}

function drawAttention(canvas, trace, layer, head) {
  const { T, attnWeights } = trace;
  const att  = attnWeights[layer];
  const cell = cs(T);

  canvas.width  = T * cell;
  canvas.height = T * cell;

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const [R, G, B] = AMBER;
  const sel = activePos >= 0 && activePos < T;

  for (let r = 0; r < T; r++) {
    for (let c = 0; c < T; c++) {
      const v = att[head * T * T + r * T + c];
      if (v < 1e-6) continue;
      const base = Math.pow(v, 0.45);
      let alpha = base;
      if (sel) {
        if      (r === activePos) alpha = base;          // selected row: full
        else if (c === activePos) alpha = base * 0.5;   // selected col: half
        else                      alpha = base * 0.15;  // everything else: dim
      }
      ctx.fillStyle = `rgba(${R},${G},${B},${alpha.toFixed(3)})`;
      ctx.fillRect(c * cell, r * cell, cell - 1, cell - 1);
    }
  }

  // Row and column borders for selected token — drawn after cells so they sit on top
  if (activePos >= 0 && activePos < T) {
    ctx.strokeStyle = 'rgba(200,146,42,0.95)';
    ctx.lineWidth   = 1.5;
    // Row: selected token as query (what it attends to)
    ctx.strokeRect(0.5, activePos * cell + 0.5, T * cell - 1, cell - 1);
    // Column: selected token as key (what attends to it)
    ctx.strokeRect(activePos * cell + 0.5, 0.5, cell - 1, T * cell - 1);
  }

  // Attach hover tooltip
  const tip = $('attn-tip');
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const col  = Math.floor((e.clientX - rect.left) / rect.width  * T);
    const row  = Math.floor((e.clientY - rect.top)  / rect.height * T);
    if (row >= 0 && row < T && col >= 0 && col < T) {
      const v    = att[head * T * T + row * T + col];
      const qTok = trace.tokenStrings[row].replace(/ /g, '·');
      const kTok = trace.tokenStrings[col].replace(/ /g, '·');
      tip.innerHTML = `${qTok} → ${kTok}<span class="tip-val">${v.toFixed(3)}</span>`;
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY + 14) + 'px';
    }
  };
  canvas.onmouseleave = () => { tip.style.display = 'none'; };
}

function drawResiduals(canvas, trace) {
  const { T, nLayer, nEmbd, embedding, residuals, final: finalNorm } = trace;
  const stages  = [embedding, ...residuals, finalNorm];
  const nStages = stages.length;                          // nLayer + 2

  const norms = new Float32Array(T * nStages);
  let maxNorm = 0;
  for (let s = 0; s < nStages; s++) {
    for (let p = 0; p < T; p++) {
      let sum = 0;
      const base = p * nEmbd;
      for (let d = 0; d < nEmbd; d++) { const v = stages[s][base + d]; sum += v * v; }
      const norm = Math.sqrt(sum);
      norms[p * nStages + s] = norm;
      if (norm > maxNorm) maxNorm = norm;
    }
  }

  const cH = Math.max(8, Math.min(36, Math.floor(360 / T)));
  const cW = Math.max(8, Math.min(32, Math.floor(700 / nStages)));

  canvas.width  = nStages * cW;
  canvas.height = T * cH;

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const [R, G, B] = TEAL;
  for (let p = 0; p < T; p++) {
    for (let s = 0; s < nStages; s++) {
      const v     = norms[p * nStages + s] / maxNorm;
      const alpha = Math.pow(v, 0.5).toFixed(3);
      ctx.fillStyle = `rgba(${R},${G},${B},${alpha})`;
      ctx.fillRect(s * cW, p * cH, cW - 1, cH - 1);
    }
  }

  // Stage labels (emb, L0..Ln, fin) below the canvas
  const labelEl = document.getElementById('residual-labels');
  if (labelEl) {
    const labels = ['emb', ...Array.from({ length: nLayer }, (_, i) => `L${i}`), 'fin'];
    labelEl.style.gridTemplateColumns = labels.map(() => `${cW}px`).join(' ');
    labelEl.innerHTML = labels.map(l => `<span>${l}</span>`).join('');
    labelEl.style.marginLeft = '0';
  }
}

function computeTopK(logitsData, pos, vocabSize, k) {
  const base  = pos * vocabSize;
  let max = -Infinity;
  for (let i = 0; i < vocabSize; i++) if (logitsData[base + i] > max) max = logitsData[base + i];

  let sum = 0;
  const probs = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) { probs[i] = Math.exp(logitsData[base + i] - max); sum += probs[i]; }
  for (let i = 0; i < vocabSize; i++) probs[i] /= sum;

  const indices = Array.from({ length: vocabSize }, (_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);
  return indices.slice(0, k).map(i => ({ id: i, prob: probs[i] }));
}

// ── Logit lens ────────────────────────────────────────────────────────────────

/** Return { id, prob } for the argmax token at position pos in a (T, vocab) flat array. */
function topOne(data, pos, vocabSize) {
  const base = pos * vocabSize;
  let max = -Infinity, topIdx = 0;
  for (let i = 0; i < vocabSize; i++) {
    if (data[base + i] > max) { max = data[base + i]; topIdx = i; }
  }
  let sum = 0;
  for (let i = 0; i < vocabSize; i++) sum += Math.exp(data[base + i] - max);
  const prob = 1 / sum;   // exp(0) / sum  since max term is always exp(0)
  return { id: topIdx, prob };
}

function renderLogitLens(trace) {
  const panel = $('logit-lens');
  if (!panel) return;
  if (!trace.logitLens || trace.logitLens[0] === null) {
    panel.innerHTML = '<span style="font-size:.72rem;color:var(--muted)">Re-export the model to enable this view: uv run python export_onnx_trace.py</span>';
    return;
  }

  const { T, nLayer, vocabSize, logitLens, logits, tokenStrings } = trace;

  // Columns: L0..L{n-1}, then the final logits
  const allCols   = [...logitLens, logits];
  const colLabels = [...Array.from({ length: nLayer }, (_, i) => `L${i}`), 'fin'];

  const CW = 64, CH = 28, LW = 68;
  const totalW = LW + allCols.length * (CW + 2);

  let html = `<div class="ll-wrap" style="width:${totalW}px">`;

  // Header row
  html += `<div class="ll-row"><span class="ll-corner"></span>`;
  for (const label of colLabels) html += `<span class="ll-hdr">${label}</span>`;
  html += `</div>`;

  // Data rows
  for (let pos = 0; pos < T; pos++) {
    const rowTok = tokenStrings[pos].replace(/ /g, '·');
    html += `<div class="ll-row"><span class="ll-rowlbl" title="pos ${pos}">${rowTok}</span>`;

    for (let col = 0; col < allCols.length; col++) {
      const { id, prob } = topOne(allCols[col], pos, vocabSize);
      const tok   = tokenizer.decode([id]).replace(/ /g, '·');
      const alpha = Math.pow(prob, 0.45).toFixed(3);
      const dark  = prob > 0.25;
      html +=
        `<span class="ll-cell" ` +
        `style="background:rgba(200,146,42,${alpha});color:${dark ? '#111' : 'var(--text)'}" ` +
        `title="p=${(prob * 100).toFixed(1)}% — '${tok}'">${tok}</span>`;
    }
    html += `</div>`;
  }

  html += `</div>`;
  panel.innerHTML = html;
}

// ── Attention entropy ─────────────────────────────────────────────────────────

/** Entropy of each head's attention distribution per query position (bits). */
function computeEntropy(attnData, T, nHead) {
  const out = new Float32Array(nHead * T);
  for (let h = 0; h < nHead; h++) {
    for (let r = 0; r < T; r++) {
      let H = 0;
      for (let c = 0; c <= r; c++) {
        const p = attnData[h * T * T + r * T + c];
        if (p > 1e-9) H -= p * Math.log2(p);
      }
      out[h * T + r] = H;
    }
  }
  return out;
}

function renderEntropyGrid(trace) {
  const panel = $('entropy-grid');
  if (!panel) return;

  const { T, nLayer, nHead, attnWeights } = trace;

  // nHead rows × nLayer columns of entropy heatmaps (one scalar per query pos)
  // Flatten to: row = head, col = layer; colour = mean entropy across positions
  // — actually richer: show per-position entropy as a mini-strip inside each cell

  const cW = Math.max(6, Math.min(32, Math.floor(600 / (nLayer * T))));
  const cH = 18;

  const canvas = document.createElement('canvas');
  canvas.width  = nLayer * (T * cW + 4);
  canvas.height = nHead * (cH + 4);
  canvas.style.imageRendering = 'pixelated';

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Global max entropy for normalisation
  let maxH = 0;
  const allEntropies = attnWeights.map(aw => {
    const e = computeEntropy(aw, T, nHead);
    for (let i = 0; i < e.length; i++) if (e[i] > maxH) maxH = e[i];
    return e;
  });

  const [R, G, B] = AMBER;
  for (let layer = 0; layer < nLayer; layer++) {
    const ent = allEntropies[layer];
    for (let h = 0; h < nHead; h++) {
      const x0 = layer * (T * cW + 4);
      const y0 = h * (cH + 4);
      for (let pos = 0; pos < T; pos++) {
        const v     = maxH > 0 ? ent[h * T + pos] / maxH : 0;
        const alpha = Math.pow(v, 0.5).toFixed(3);
        ctx.fillStyle = `rgba(${R},${G},${B},${alpha})`;
        ctx.fillRect(x0 + pos * cW, y0, cW - 1, cH);
      }
    }
  }

  // Labels: layer below, head to the left — keep them in HTML above/beside canvas
  const labelEl = $('entropy-labels');
  if (labelEl) {
    labelEl.innerHTML =
      `<span style="font-size:.65rem;color:var(--muted)">` +
      `rows = heads 0‥${nHead - 1} · cols = layers 0‥${nLayer - 1} · ` +
      `each strip = token positions · brightness = attention entropy` +
      `</span>`;
  }

  panel.innerHTML = '';
  panel.appendChild(canvas);
}

// ── Rendering ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function renderTokens(trace) {
  const strip = $('tokens');
  strip.innerHTML = '';
  trace.tokenStrings.forEach((tok, i) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip' + (i === activePos ? ' active' : '');
    chip.textContent = tok.replace(/ /g, '·');   // space → ·
    chip.title = `pos ${i}  id ${trace.tokenIds[i]}`;
    chip.addEventListener('click', () => {
      activePos = activePos === i ? -1 : i;
      renderAll(trace);
    });
    strip.appendChild(chip);
  });
}

function renderLayerTabs(trace) {
  const tabs = $('layer-tabs');
  tabs.innerHTML = '';
  for (let i = 0; i < trace.nLayer; i++) {
    const btn = document.createElement('button');
    btn.className = 'layer-tab' + (i === activeLayer ? ' active' : '');
    btn.textContent = `L${i}`;
    btn.addEventListener('click', () => { activeLayer = i; renderAll(trace); });
    tabs.appendChild(btn);
  }
}

function headLabel(h, nHead) {
  // Assumes layout: 1 large (64d) + 2 med (32d) + 8 small (8d)
  if (h === 0)        return 'Semantic 64d';
  if (h <= 2)         return `Context ${h} · 32d`;
  return `Syntax ${h - 3} · 8d`;
}

function renderHeads(trace) {
  const grid = $('head-grid');
  // Rebuild canvases if head count changed (first run or model swap)
  if (grid.children.length !== trace.nHead) {
    grid.innerHTML = '';
    for (let h = 0; h < trace.nHead; h++) {
      const wrap   = document.createElement('div');
      wrap.className = 'head-wrap';
      const label  = document.createElement('span');
      label.className = 'head-label';
      label.textContent = headLabel(h, trace.nHead);
      const canvas = document.createElement('canvas');
      canvas.id = `attn-${h}`;
      wrap.append(label, canvas);
      grid.appendChild(wrap);
    }
  }
  for (let h = 0; h < trace.nHead; h++) {
    drawAttention($(`attn-${h}`), trace, activeLayer, h);
  }
}

function renderTopK(trace) {
  const panel = $('topk');
  if (!panel) return;

  if (activePos < 0) {
    panel.innerHTML = '';
    return;
  }

  const items = computeTopK(trace.logits, activePos, trace.vocabSize, 10);
  const rows  = items.map(({ id, prob }) => {
    const tok = tokenizer.decode([id]).replace(/ /g, '·');
    return `<span class="topk-token">${tok}</span><span class="topk-prob">${(prob * 100).toFixed(1)}%</span>`;
  }).join('');

  panel.innerHTML =
    `<div class="section-label">Top predictions after pos ${activePos} ` +
    `<span class="muted">("${trace.tokenStrings[activePos]}")</span></div>` +
    `<div class="topk-list">${rows}</div>`;
}

function renderAll(trace) {
  renderTokens(trace);
  renderLayerTabs(trace);
  renderHeads(trace);
  drawResiduals($('residual-canvas'), trace);
  renderTopK(trace);
  renderLogitLens(trace);
  renderEntropyGrid(trace);
  $('viz').style.display = '';
}

// ── UI ────────────────────────────────────────────────────────────────────────

async function onTrace() {
  const prompt = $('prompt').value.trim();
  if (!prompt || !session) return;

  $('trace-btn').disabled = true;
  $('status').textContent = 'Running…';

  try {
    activePos  = -1;
    traceData  = await runTrace(prompt);
    renderAll(traceData);
    $('status').textContent =
      `${traceData.T} token${traceData.T !== 1 ? 's' : ''} · ${traceData.nLayer} layers · ${traceData.nHead} heads`;
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    $('trace-btn').disabled = false;
  }
}

async function init() {
  $('status').textContent = 'Loading model…';
  try {
    await loadModels();
    $('status').textContent = 'Ready.';
    $('trace-btn').disabled = false;
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  $('trace-btn').addEventListener('click', onTrace);

  // Wire example prompt buttons
  const exampleBtns = document.querySelectorAll('.example-btn');
  const hintBox = document.getElementById('example-hint');
  exampleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const text = btn.dataset.text;
      $('prompt').value = text;
      if (hintBox && btn.dataset.hint) hintBox.textContent = btn.dataset.hint;
      onTrace();
    });
  });

  init();
});
