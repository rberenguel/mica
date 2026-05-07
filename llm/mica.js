import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';
import { loadTokenizer } from './tokenizer.js';

const BLOCK_SIZE = 256;

// ── Sampling ──────────────────────────────────────────────────────────────────

function sampleLogits(logits, temperature) {
  const n = logits.length;
  const scaled = new Float32Array(n);

  // Temperature scaling + find max for numerical stability
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    scaled[i] = logits[i] / temperature;
    if (scaled[i] > max) max = scaled[i];
  }

  // Softmax
  let sum = 0;
  for (let i = 0; i < n; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }

  // Multinomial sample
  const threshold = Math.random() * sum;
  let cumulative = 0;
  for (let i = 0; i < n; i++) {
    cumulative += scaled[i];
    if (cumulative >= threshold) return i;
  }
  return n - 1;
}

// ── Model ─────────────────────────────────────────────────────────────────────

async function loadModel() {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  return ort.InferenceSession.create('./mica_trace.onnx', {
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
  // trace model returns logits (1, T, vocab_size); take last position
  const logits   = result.logits.data;
  const vocabSize = logits.length / T;
  return logits.slice((T - 1) * vocabSize, T * vocabSize);
}

// ── Generation ────────────────────────────────────────────────────────────────

/**
 * Async generator: yields decoded token strings one at a time.
 * Caller can break out of the loop to stop generation.
 */
async function* generate(session, tokenizer, prompt, temperature, maxTokens) {
  let ids = tokenizer.encode(prompt);
  if (ids.length === 0) ids = [0]; // fallback: start token

  for (let i = 0; i < maxTokens; i++) {
    const logits = await runModel(session, ids);
    const nextId = sampleLogits(logits, temperature);
    ids.push(nextId);
    yield tokenizer.decode([nextId]);
  }
}

// ── UI ────────────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

let session = null;
let tokenizer = null;
let stopRequested = false;

async function init() {
  $('status').textContent = 'Loading model…';
  try {
    [session, tokenizer] = await Promise.all([loadModel(), loadTokenizer()]);
    $('status').textContent = 'Ready.';
    $('generate-btn').disabled = false;
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

async function onGenerate() {
  if (!session || !tokenizer) return;

  const prompt      = $('prompt').value.trim();
  const temperature = parseFloat($('temperature').value);
  const maxTokens   = parseInt($('max-tokens').value, 10);

  if (!prompt) return;

  $('generate-btn').disabled = true;
  $('stop-btn').disabled = false;
  $('output').textContent = prompt;
  $('status').textContent = 'Generating…';
  stopRequested = false;

  try {
    for await (const token of generate(session, tokenizer, prompt, temperature, maxTokens)) {
      if (stopRequested) break;
      $('output').textContent += token;
      // Keep the output scrolled to bottom
      $('output').scrollTop = $('output').scrollHeight;
    }
    $('status').textContent = stopRequested ? 'Stopped.' : 'Done.';
  } catch (err) {
    $('status').textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    $('generate-btn').disabled = false;
    $('stop-btn').disabled = true;
  }
}

function onStop() {
  stopRequested = true;
}

function onTempChange() {
  $('temperature-display').textContent = parseFloat($('temperature').value).toFixed(2);
}

document.addEventListener('DOMContentLoaded', () => {
  $('generate-btn').addEventListener('click', onGenerate);
  $('stop-btn').addEventListener('click', onStop);
  $('temperature').addEventListener('input', onTempChange);
  init();
});
