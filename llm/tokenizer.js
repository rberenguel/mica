/**
 * Minimal ByteLevel BPE tokenizer for Mica.
 * Reads the HuggingFace tokenizer JSON format produced by prepare.py.
 *
 * Limitations: pre-tokenizer splits on ASCII spaces only.
 * Contractions and unicode edge-cases may differ from the Python tokenizer.
 */

function buildByteToUnicode() {
  // Mirrors the Python bytes_to_unicode() function used by HuggingFace / GPT-2.
  // Bytes 33-126, 161-172, 174-255 map to themselves (printable Latin-1).
  // All other bytes map to U+0100 onwards in sorted order.
  // Critically: byte 32 (space) → U+0120 (Ġ).
  const encode = new Uint16Array(256);
  const inRange = (b) =>
    (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);

  let n = 0;
  for (let b = 0; b < 256; b++) {
    encode[b] = inRange(b) ? b : 256 + n++;
  }
  return encode;
}

const BYTE_ENC = buildByteToUnicode();

// Inverse table: unicode char → byte value
const BYTE_DEC = {};
for (let b = 0; b < 256; b++) {
  BYTE_DEC[String.fromCharCode(BYTE_ENC[b])] = b;
}

/** Encode a raw string to byte-level unicode (Ġ for space, etc.) */
function textToByteLevel(text) {
  const utf8 = new TextEncoder().encode(text);
  return Array.from(utf8, (b) => String.fromCharCode(BYTE_ENC[b])).join('');
}

/** Decode a byte-level unicode string back to UTF-8 text */
function byteLevelToText(blText) {
  const bytes = Array.from(blText, (c) => BYTE_DEC[c]).filter((b) => b !== undefined);
  return new TextDecoder().decode(new Uint8Array(bytes));
}

export class MicaTokenizer {
  /**
   * @param {object} json — parsed mica_tokenizer.json
   */
  constructor(json) {
    this.vocab = json.model.vocab;                 // token string → id
    this.idToToken = Object.fromEntries(
      Object.entries(this.vocab).map(([k, v]) => [v, k])
    );
    // Merge rules stored as a Map: "a b" → priority index (lower = higher priority)
    // json.model.merges is an array of [a, b] pairs; join with space to match lookup format
    this.merges = new Map(json.model.merges.map(([a, b], i) => [a + ' ' + b, i]));
  }

  /** Apply BPE merges to a single pre-tokenized word (byte-level encoded). */
  _bpe(word) {
    let tokens = [...word]; // split into individual unicode chars (code points)
    if (tokens.length <= 1) return tokens;

    while (true) {
      let bestPriority = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = tokens[i] + ' ' + tokens[i + 1];
        const priority = this.merges.get(pair);
        if (priority !== undefined && priority < bestPriority) {
          bestPriority = priority;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      const merged = tokens[bestIdx] + tokens[bestIdx + 1];
      tokens.splice(bestIdx, 2, merged);
    }
    return tokens;
  }

  /**
   * Encode a string to an array of token IDs.
   * @param {string} text
   * @returns {number[]}
   */
  encode(text) {
    if (!text) return [];

    // ByteLevel pre-tokenization: split on spaces, prepend Ġ to all but first word.
    const words = [];
    let i = 0;
    while (i < text.length) {
      let word = '';
      // Consume a run of non-space characters
      while (i < text.length && text[i] !== ' ') {
        word += text[i++];
      }
      if (word) {
        const prefix = words.length > 0 ? 'Ġ' : ''; // Ġ = space in byte-level
        words.push(prefix + textToByteLevel(word));
      }
      // Skip spaces (they're absorbed into the Ġ prefix of the next word)
      while (i < text.length && text[i] === ' ') i++;
    }

    const ids = [];
    for (const word of words) {
      for (const token of this._bpe(word)) {
        const id = this.vocab[token];
        if (id !== undefined) ids.push(id);
        // Unknown tokens are silently skipped — acceptable for tinkering
      }
    }
    return ids;
  }

  /**
   * Decode an array of token IDs back to a string.
   * @param {number[]} ids
   * @returns {string}
   */
  decode(ids) {
    const byteLevel = ids.map((id) => this.idToToken[id] ?? '').join('');
    return byteLevelToText(byteLevel);
  }
}

/** Load and construct a MicaTokenizer from a URL. */
export async function loadTokenizer(url = './mica_tokenizer.json') {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load tokenizer: ${resp.status} ${url}`);
  const json = await resp.json();
  return new MicaTokenizer(json);
}
