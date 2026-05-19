import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs';
import { loadTokenizer } from './tokenizer.js';

const BLOCK_SIZE = 512;

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

// ── Post-processing (mirrors generate_origami.py _postprocess) ──────────────

function postprocess(text) {
  if (!text) return text;

  // 1. Capitalise first character
  text = text[0].toUpperCase() + text.slice(1);

  // 2. Capitalise after sentence boundaries
  text = text.replace(/([.!?] +)([a-z])/g, (_, boundary, letter) => boundary + letter.toUpperCase());
  text = text.replace(/(\n)([a-z])/g, (_, newline, letter) => newline + letter.toUpperCase());

  // 3. Insert missing space after punctuation before next word
  text = text.replace(/([.!?;,])([A-Za-z])/g, '$1 $2');

  // 4. Fix common stuck-together BPE words
  text = text.replace(/\bThe(man|office|room|hall|door|boy|gun|car|street|house|hotel)\b/gi, 'The $1');
  text = text.replace(/\bA(man|boy|girl|gun|door|room|car|bullet|knife)\b/gi, 'A $1');
  text = text.replace(/\b(Man|Boy|Girl|Gun|Door)([a-z]+)\b/g, '$1 $2');

  // 5. Collapse multiple spaces
  text = text.replace(/ +/g, ' ');

  // 6. Strip isolated number garbage at line edges
  text = text.replace(/^\d+\.? */gm, '');
  text = text.replace(/ *\d+\.?$/gm, '');

  // 7. Strip underscores (OCR artifacts)
  text = text.replace(/_/g, ' ');

  return text;
}

// ── Generation ────────────────────────────────────────────────────────────────

/**
 * Async generator: yields decoded text chunks.
 * Decodes the full accumulated sequence each step so BPE merging is
 * correct, then yields only the newly added text since the last step.
 */
async function* generate(session, tokenizer, prompt, temperature, maxTokens, minTokens = 5) {
  const eotId = 0;  // <|endoftext|>
  const eosId = 1;  // <|eos|>

  let ids = tokenizer.encode(prompt);
  if (ids.length === 0) ids = [0]; // fallback: start token

  let prevText = tokenizer.decode(ids);
  let generated = 0;

  for (let i = 0; i < maxTokens; i++) {
    const logits = await runModel(session, ids);

    // Mask out boundary tokens until we've hit the minimum length
    if (generated < minTokens) {
      logits[eosId] = -Infinity;
      logits[eotId] = -Infinity;
    }

    const nextId = sampleLogits(logits, temperature);
    generated++;

    // Natural stop on a boundary token once we're past the minimum
    if ((nextId === eotId || nextId === eosId) && generated >= minTokens) {
      break;
    }

    ids.push(nextId);

    const fullText = tokenizer.decode(ids);
    // Only yield the newly added portion
    if (fullText.length > prevText.length) {
      const newText = fullText.slice(prevText.length);
      prevText = fullText;
      yield newText;
    }
    // If decode didn't grow (rare BPE edge case), silently continue
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

  let rawText = prompt;
  try {
    for await (const token of generate(session, tokenizer, prompt, temperature, maxTokens, 5)) {
      if (stopRequested) break;
      rawText += token;
      $('output').textContent = postprocess(rawText);
      $('output').scrollTop = $('output').scrollHeight;
    }
    // Final clean pass
    $('output').textContent = postprocess(rawText);
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
