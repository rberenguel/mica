import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';
import { loadTokenizer } from './tokenizer.js';

const BLOCK_SIZE = 512;
const DEFAULT_MODEL = './mica_trace.onnx';

// ── Sampling ──────────────────────────────────────────────────────────────────

function sampleLogits(logits, temperature) {
  const n = logits.length;
  const scaled = new Float32Array(n);

  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    scaled[i] = logits[i] / temperature;
    if (scaled[i] > max) max = scaled[i];
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }

  const threshold = Math.random() * sum;
  let cumulative = 0;
  for (let i = 0; i < n; i++) {
    cumulative += scaled[i];
    if (cumulative >= threshold) return i;
  }
  return n - 1;
}

// ── Model ─────────────────────────────────────────────────────────────────────

async function loadModel(modelPath) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  return ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
  });
}

async function runModel(session, ids) {
  const seq = ids.slice(-BLOCK_SIZE);
  const T   = seq.length;
  const input = new ort.Tensor(
    'int64',
    BigInt64Array.from(seq, BigInt),
    [1, T],
  );
  const result = await session.run({ input_ids: input });
  const logits   = result.logits.data;
  const vocabSize = logits.length / T;
  return logits.slice((T - 1) * vocabSize, T * vocabSize);
}

// ── Post-processing (mirrors generate_origami.py _postprocess) ──────────────

function postprocess(text) {
  if (!text) return text;

  // 1. Capitalise first character
  text = text[0].toUpperCase() + text.slice(1);

  // 2. Capitalise after sentence-ending punctuation + space
  text = text.replace(/([.!?] +)([a-z])/g, (_, boundary, letter) => boundary + letter.toUpperCase());

  // 3. Capitalise after paragraph break (double newline) or newline preceded by .!?
  //    Don't capitalise after mid-sentence newlines — the corpus has OCR line breaks
  text = text.replace(/([.!?])\n([a-z])/g, (_, punct, letter) => punct + '\n' + letter.toUpperCase());
  text = text.replace(/\n\n([a-z])/g, (_, letter) => '\n\n' + letter.toUpperCase());

  // 4. Insert missing space after punctuation before next word
  text = text.replace(/([.!?;,])([A-Za-z])/g, '$1 $2');

  // 5. Fix common stuck-together BPE words
  text = text.replace(/\bThe(man|office|room|hall|door|boy|gun|car|street|house|hotel)\b/gi, 'The $1');
  text = text.replace(/\bA(man|boy|girl|gun|door|room|car|bullet|knife)\b/gi, 'A $1');
  text = text.replace(/\b(Man|Boy|Girl|Gun|Door)([a-z]+)\b/g, '$1 $2');

  // 6. Collapse multiple spaces
  text = text.replace(/ +/g, ' ');

  // 7. Strip isolated number garbage at line edges
  text = text.replace(/^\d+\.? */gm, '');
  text = text.replace(/ *\d+\.?$/gm, '');

  // 8. Strip underscores (OCR artifacts)
  text = text.replace(/_/g, ' ');

  return text;
}

// ── Generation ────────────────────────────────────────────────────────────────

async function* generate(session, tokenizer, prompt, temperature, maxTokens, minTokens = 5) {
  const eotId = 0;
  const eosId = 1;

  let ids = tokenizer.encode(prompt);
  if (ids.length === 0) ids = [0];

  let prevText = tokenizer.decode(ids);
  let generated = 0;

  for (let i = 0; i < maxTokens; i++) {
    const logits = await runModel(session, ids);

    if (generated < minTokens) {
      logits[eosId] = -Infinity;
      logits[eotId] = -Infinity;
    }

    const nextId = sampleLogits(logits, temperature);
    generated++;

    if ((nextId === eotId || nextId === eosId) && generated >= minTokens) {
      break;
    }

    ids.push(nextId);

    const fullText = tokenizer.decode(ids);
    if (fullText.length > prevText.length) {
      const newText = fullText.slice(prevText.length);
      prevText = fullText;
      yield newText;
    }
  }
}

// ── UI ────────────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

let session = null;
let tokenizer = null;
let stopRequested = false;
let currentModelPath = DEFAULT_MODEL;
let currentModelLabel = 'Mica (5.4M)';

async function switchModel(path, label) {
  currentModelPath = path;
  currentModelLabel = label;

  $('status').textContent = `Loading ${label}…`;
  $('generate-btn').disabled = true;

  try {
    session = await loadModel(path);
    $('status').textContent = `${label} ready.`;
    $('generate-btn').disabled = false;
    $('continue-btn').disabled = false;
  } catch (err) {
    $('status').textContent = `Error loading ${label}: ${err.message}`;
    console.error(err);
  }
}

async function init() {
  tokenizer = await loadTokenizer();
  await switchModel(DEFAULT_MODEL, 'Mica (5.4M)');
}

async function onGenerate() {
  if (!session || !tokenizer) return;

  const prompt      = $('prompt').value.trim();
  const temperature = parseFloat($('temperature').value);
  const maxTokens   = parseInt($('max-tokens').value, 10);

  if (!prompt) return;

  $('generate-btn').disabled = true;
  $('continue-btn').disabled = true;
  $('stop-btn').disabled = false;
  $('output').textContent = prompt;
  $('status').textContent = `Generating with ${currentModelLabel}…`;
  stopRequested = false;

  let rawText = prompt;
  try {
    for await (const token of generate(session, tokenizer, prompt, temperature, maxTokens, 5)) {
      if (stopRequested) break;
      rawText += token;
      $('output').textContent = postprocess(rawText);
      $('output').scrollTop = $('output').scrollHeight;
    }
    $('output').textContent = postprocess(rawText);
    $('status').textContent = stopRequested ? 'Stopped.' : `Done (${currentModelLabel}).`;
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    $('generate-btn').disabled = false;
    $('continue-btn').disabled = false;
    $('stop-btn').disabled = true;
  }
}

function onStop() {
  stopRequested = true;
}

async function onContinue() {
  if (!session || !tokenizer) return;

  const output = $('output').textContent.trim();
  if (!output) return;

  const temperature = parseFloat($('temperature').value);
  const maxTokens   = parseInt($('max-tokens').value, 10);

  $('generate-btn').disabled = true;
  $('continue-btn').disabled = true;
  $('stop-btn').disabled = false;
  $('status').textContent = `Continuing with ${currentModelLabel}…`;
  stopRequested = false;

  let rawText = output;
  try {
    for await (const token of generate(session, tokenizer, output, temperature, maxTokens, 5)) {
      if (stopRequested) break;
      rawText += token;
      $('output').textContent = postprocess(rawText);
      $('output').scrollTop = $('output').scrollHeight;
    }
    $('output').textContent = postprocess(rawText);
    $('status').textContent = stopRequested ? 'Stopped.' : `Done (${currentModelLabel}).`;
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    $('generate-btn').disabled = false;
    $('continue-btn').disabled = false;
    $('stop-btn').disabled = true;
  }
}

function onTempChange() {
  $('temperature-display').textContent = parseFloat($('temperature').value).toFixed(2);
}

document.addEventListener('DOMContentLoaded', () => {
  $('generate-btn').addEventListener('click', onGenerate);
  $('stop-btn').addEventListener('click', onStop);
  $('continue-btn').addEventListener('click', onContinue);
  $('temperature').addEventListener('input', onTempChange);

  // Wire model selector buttons
  document.querySelectorAll('.model-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.model-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      switchModel('./' + btn.dataset.model, btn.dataset.label);
    });
  });

  init();
});
