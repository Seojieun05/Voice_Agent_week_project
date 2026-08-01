// Capture worklet: runs on the browser's real-time audio thread and posts
// raw float32 sample blocks (128 samples, mono) to the main thread, where
// framing to 20 ms and float32 -> int16 conversion happen.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) {
      // Copy! The engine reuses this buffer the instant we return.
      this.port.postMessage(new Float32Array(channel));
    }
    return true; // keep the processor alive
  }
}

registerProcessor("pcm-capture", PCMCaptureProcessor);
