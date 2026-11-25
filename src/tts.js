// tiny-voca Japanese の前処理 + ONNX 推論を JS に移植したもの。
// 移植元: tiny-voca/src/tiny_voca/japanese.py (from_labels), japanese_text.py, japanese_onnx.py (speak_text)

export const SPECIALS = ['<pad>', '<bos>', '<eos>', '<sp>', '<unk>'];
export const PHONES = ['a', 'i', 'u', 'e', 'o', 'A', 'I', 'U', 'E', 'O', 'N', 'cl',
  'p', 'py', 'b', 'by', 'm', 'my', 't', 'ty', 'd', 'dy', 'n', 'ny',
  'k', 'ky', 'kw', 'g', 'gy', 'gw', 's', 'sh', 'z', 'j', 'ch', 'ts',
  'f', 'h', 'hy', 'v', 'r', 'ry', 'w', 'y'];
const SYMBOLS = [...SPECIALS, ...PHONES];
const SYMBOL_TO_ID = Object.fromEntries(SYMBOLS.map((s, i) => [s, i]));
const PHONE_SET = new Set(PHONES);
const MISSING_ACCENT = [-50, 0, 0];
const BOUNDARY_PHONES = new Set([...'aiueoAIUEON', 'cl']);

// ---- japanese_text.py ----
const CLOSING = "\"'”’)]»」』）】〉》";

function rstripChars(s, chars) {
  let end = s.length;
  while (end > 0 && chars.includes(s[end - 1])) end--;
  return s.slice(0, end);
}

export function japaneseFinished(text) {
  const body = text.replace(/[\s﻿]+$/u, '');
  const end = rstripChars(body, CLOSING);
  return !end || '。！？.!?…、，,;:；：'.includes(end[end.length - 1]) ? text : body + '。';
}

export function japaneseSentences(text) {
  text = text.replace(/[\s﻿]+/gu, ' ').trim();
  const parts = [];
  let start = 0;
  for (const m of text.matchAll(/[。！？!?]+[」』）】〉》”’"')\]]*/gu)) {
    parts.push(text.slice(start, m.index + m[0].length).trim());
    start = m.index + m[0].length;
  }
  parts.push(text.slice(start).trim());
  return parts.filter((p) => /[\p{L}\p{N}]/u.test(p));
}

export function japaneseShorter(text) {
  const parts = text.split(/(?<=[、，,;；：:])/u).map((p) => p.trim()).filter(Boolean);
  if (parts.length > 1) return parts;
  const chars = [...text];
  if (chars.length < 2) return [];
  const half = Math.floor((chars.length + 1) / 2);
  return [chars.slice(0, half).join(''), chars.slice(half).join('')];
}

// ---- japanese.py ----
export function normalize(text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('Japanese text must not be empty');
  return text.normalize('NFC').trim();
}

export function fromLabels(labels) {
  const parsed = [];
  for (const lab of labels) {
    const m = lab.match(/-(.*?)\+.*?\/A:([^/]+)/);
    if (!m) throw new Error('malformed full-context label');
    const [, phone, raw] = m;
    let accent;
    if (phone === 'sil' || phone === 'pau') {
      accent = MISSING_ACCENT;
    } else {
      if (!PHONE_SET.has(phone)) throw new Error(`unsupported Japanese phone: ${phone}`);
      accent = raw.split('+').map((v) => {
        if (!/^-?\d+$/.test(v)) throw new Error('missing Japanese accent fields');
        return parseInt(v, 10);
      });
      if (accent.length !== 3 || !(accent[0] >= -49 && accent[0] <= 49) || !accent.slice(1).every((v) => v >= 1 && v <= 49)) {
        throw new Error('invalid Japanese accent fields');
      }
    }
    parsed.push([phone, accent]);
  }
  if (parsed.length < 3 || parsed[0][0] !== 'sil' || parsed[parsed.length - 1][0] !== 'sil') {
    throw new Error('Japanese labels need speech between utterance edges');
  }
  if (parsed.slice(1, -1).some(([p]) => p === 'sil') || !parsed.some(([p]) => PHONE_SET.has(p))) {
    throw new Error('unexpected silence or no speech in Japanese labels');
  }
  const symbols = [], accents = [], boundaries = [];
  parsed.forEach(([phone, accent], i) => {
    const symbol = i === 0 ? '<bos>' : i === parsed.length - 1 ? '<eos>' : phone === 'pau' ? '<sp>' : phone;
    let boundary = 0;
    if (BOUNDARY_PHONES.has(phone)) {
      const [following, nextAccent] = parsed[i + 1];
      boundary = Number(following === 'sil' || following === 'pau' || (accent[2] === 1 && nextAccent[1] === 1));
    }
    symbols.push(symbol);
    accents.push(accent);
    boundaries.push(boundary);
  });
  return { symbols, ids: symbols.map((s) => SYMBOL_TO_ID[s]), accent: accents, phraseBoundary: boundaries };
}

export function phonemize(jpreprocess, text) {
  return fromLabels(jpreprocess.extract_fullcontext(normalize(text)));
}

// ---- japanese_onnx.py (speak_text) ----
export async function speakText(ort, session, jpreprocess, meta, text, { speed = 1, pitchShift = 0, pitchRange = 1, peak = 0.99 } = {}) {
  const waves = [];
  const spoken = [];
  const voice = 0;

  async function read(piece) {
    const phones = phonemize(jpreprocess, piece);
    const n = phones.ids.length;
    if (n <= meta.max_tokens - 12) {
      const feeds = {
        tokens: new ort.Tensor('int64', BigInt64Array.from(phones.ids, BigInt), [1, n]),
        accent: new ort.Tensor('int64', BigInt64Array.from(phones.accent.flat(), BigInt), [1, n, 3]),
        phrase_boundary: new ort.Tensor('int64', BigInt64Array.from(phones.phraseBoundary, BigInt), [1, n]),
        voice: new ort.Tensor('int64', BigInt64Array.from([voice], BigInt), [1]),
        speed: new ort.Tensor('float32', Float32Array.from([speed]), [1]),
        pitch_shift: new ort.Tensor('float32', Float32Array.from([pitchShift]), [1]),
        pitch_range: new ort.Tensor('float32', Float32Array.from([pitchRange]), [1]),
      };
      const out = await session.run(feeds);
      const frames = Array.from(out.durations.data, Number).reduce((a, b) => a + b, 0);
      if (frames <= meta.max_frames) {
        const audio = out.audio.data;
        if (audio.length !== frames * meta.hop || !audio.every(Number.isFinite)) {
          throw new Error('ONNX produced invalid or inconsistent audio');
        }
        waves.push(audio);
        spoken.push(phones);
        return;
      }
    }
    const parts = japaneseShorter(piece);
    if (!parts.length) throw new Error('one character exceeds the synthesis limit');
    for (const part of parts) await read(part);
  }

  for (const piece of japaneseSentences(japaneseFinished(text))) await read(piece);
  if (!waves.length) throw new Error('日本語の読み上げ対象がありません');

  const wave = new Float32Array(waves.reduce((a, w) => a + w.length, 0));
  let offset = 0;
  for (const w of waves) { wave.set(w, offset); offset += w.length; }
  let loudest = 0;
  for (const v of wave) loudest = Math.max(loudest, Math.abs(v));
  if (peak != null && loudest > peak) for (let i = 0; i < wave.length; i++) wave[i] *= peak / loudest;
  return { wave, spoken };
}

export const DICT_FILES = {
  dict_trie: 'dict.trie', dict_idx: 'dict.valsidx', dict_vals: 'dict.vals', cost_matrix: 'matrix.mtx',
  char_definitions: 'char_def.bin', unknown_dictionary: 'unk.bin', words_idx_data: 'dict.wordsidx', words_data: 'dict.words',
};
