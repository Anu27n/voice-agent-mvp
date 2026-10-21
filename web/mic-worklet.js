// Captures microphone audio, downsamples it to the server's input rate and
// posts fixed-size PCM16 frames (plus the frame's level in dBFS) to the page.
class MicCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { targetRate, frameMs } = options.processorOptions;
    this.step = sampleRate / targetRate; // input samples per output sample
    this.frameSamples = Math.round((targetRate * frameMs) / 1000);
    this.frame = new Int16Array(this.frameSamples);
    this.filled = 0;
    this.energy = 0;
    this.acc = 0; // running sum of input samples for the current output sample
    this.count = 0;
    this.phase = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      // Box-filter decimation: average the input samples that fall into each
      // output sample. Cheap, and enough of a low-pass for speech recognition.
      this.acc += channel[i];
      this.count += 1;
      this.phase += 1;
      if (this.phase >= this.step) {
        this.phase -= this.step;
        this.push(this.acc / this.count);
        this.acc = 0;
        this.count = 0;
      }
    }
    return true;
  }

  push(value) {
    const v = Math.max(-1, Math.min(1, value));
    this.frame[this.filled++] = v < 0 ? v * 32768 : v * 32767;
    this.energy += v * v;
    if (this.filled === this.frameSamples) {
      const rms = Math.sqrt(this.energy / this.frameSamples);
      const db = rms > 0 ? 20 * Math.log10(rms) : -100;
      this.port.postMessage({ pcm: this.frame.buffer, db }, [this.frame.buffer]);
      this.frame = new Int16Array(this.frameSamples);
      this.filled = 0;
      this.energy = 0;
    }
  }
}

registerProcessor("mic-capture", MicCapture);
