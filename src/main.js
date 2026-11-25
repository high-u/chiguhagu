import Alpine from 'alpinejs';
import { WORDS } from './words.js';

const MODEL = 'https://huggingface.co/ayousanz/tiny-voca-ja-tsukuyomi-chan/resolve/main/models/onnx/';

let ort, tts, jp, session, meta, audioCtx, clearTimer;

const bytes = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
const json = async (url) => (await fetch(url)).json();
const pick = (list) => list[Math.floor(Math.random() * list.length)];

// {いつ}、{どこ}で、{誰}が、{何}を、{どうした}らしいよ。
function makeSentence() {
  const itsu = pick(WORDS.いつ);
  const verb = pick(WORDS.どうした)[itsu.文末];
  return `${itsu.語}、${pick(WORDS.どこ)}で、${pick(WORDS.誰が)}が、${pick(WORDS.何を)}を、${verb}らしいよ。`;
}

// 音声のレベルから、口の開き（1 = 閉じている、3 = 最大）を決める。-50dB 以下は閉じ、-30dB で最大
function mouthScale(samples) {
  const rms = Math.sqrt(samples.reduce((sum, v) => sum + v * v, 0) / samples.length);
  const open = (20 * Math.log10(rms) + 50) / 20;
  return 1 + 2 * Math.min(Math.max(open, 0), 1);
}

async function loadAll() {
  ort = await import('onnxruntime-web/wasm');
  tts = await import('./tts.js');
  const { default: init, JPreprocess } = await import('./jpreprocess/example_wasm.js');

  await init();
  const dict = { metadata: await json('./dict/metadata.json') };
  for (const [key, file] of Object.entries(tts.DICT_FILES)) dict[key] = await bytes('./dict/' + file);
  jp = new JPreprocess(dict);

  ort.env.wasm.numThreads = 1;
  meta = await json(MODEL + 'model.json');
  session = await ort.InferenceSession.create(MODEL + 'model.onnx', {
    externalData: [{ path: 'model.weights.bin', data: MODEL + 'model.weights.bin' }],
  });
}

Alpine.data('app', () => ({
  started: false,
  ready: false, // 「何か話して」を押せるか
  mouth: 1, // 口の縦の倍率
  sentence: '', // 表示する文章

  async start() {
    this.started = true;
    audioCtx = new AudioContext();
    await loadAll();
    this.ready = true;
  },

  async speak() {
    this.ready = false;
    clearTimeout(clearTimer);
    const sentence = makeSentence();
    const { wave } = await tts.speakText(ort, session, jp, meta, sentence, { pitchRange: 0.5 });
    const buffer = audioCtx.createBuffer(1, wave.length, meta.sample_rate);
    buffer.copyToChannel(wave, 0);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    const analyser = audioCtx.createAnalyser();
    const volume = audioCtx.createGain();
    volume.gain.value = 1.25;
    source.connect(analyser);
    analyser.connect(volume);
    volume.connect(audioCtx.destination);

    const samples = new Float32Array(analyser.fftSize);
    let playing = true;
    const move = () => {
      if (!playing) return;
      analyser.getFloatTimeDomainData(samples);
      this.mouth = mouthScale(samples);
      requestAnimationFrame(move);
    };
    source.onended = () => {
      playing = false;
      this.mouth = 1;
      this.ready = true;
      clearTimer = setTimeout(() => { this.sentence = ''; }, 1000);
    };
    source.start();
    this.sentence = sentence;
    move();
  },
}));

Alpine.start();
