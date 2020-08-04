import * as onnxjs from 'onnxjs';
import * as onnxruntime from 'onnxruntime';

export class OnnxRuntimeInferenceSession implements onnxjs.InferenceSession {
  private onnxjsFallback?: onnxjs.InferenceSession;
  private binding?: onnxruntime.InferenceSession;

  constructor(config?: onnxjs.InferenceSession.Config) {
    let useOnnxRuntime = !config || typeof config.backendHint !== 'string' || config.backendHint === 'onnxruntime';

    if (!useOnnxRuntime) {
      this.onnxjsFallback = new onnxjs.InferenceSession(config);
      console.log('fallback to ONNX.js inference session');
    }
  }

  loadModel(uri: string): Promise<void>;
  loadModel(blob: Blob): Promise<void>;
  loadModel(buffer: ArrayBuffer|SharedArrayBuffer, byteOffset?: number, length?: number): Promise<void>;
  loadModel(buffer: Uint8Array): Promise<void>;
  async loadModel(arg0: any, arg1?: any, arg2?: any): Promise<void> {
    if (this.onnxjsFallback) {
      return this.onnxjsFallback.loadModel(arg0, arg1, arg2);
    }

    if (typeof arg0 !== 'string') {
      throw new TypeError('a string model path is expected');
    }

    this.binding = await onnxruntime.InferenceSession.create(arg0);
  }

  async run(inputFeed: onnxjs.InferenceSession.InputType, options?: onnxjs.InferenceSession.RunOptions):
      Promise<ReadonlyMap<string, onnxjs.Tensor>> {
    if (this.onnxjsFallback) {
      return this.onnxjsFallback.run(inputFeed, options);
    }

    if (!this.binding) {
      throw new Error('session not initialized');
    }

    const input = {};
    if (inputFeed instanceof Map) {
      this.binding.inputNames.forEach((name) => {
        const t = inputFeed.get(name);
        if (!t) {
          throw new Error(`missing input '${name}'`);
        }
        input[name] = new onnxruntime.Tensor(t.type, t.data, t.dims);
      });
    } else if (Array.isArray(inputFeed)) {
      inputFeed.forEach((t, i) => {
        input[this.binding!.inputNames[i]] = new onnxruntime.Tensor(t.type, t.data, t.dims);
      });
    } else {
      this.binding.inputNames.forEach((name) => {
        const t = (inputFeed as {readonly [name: string]: any})[name];
        if (!t) {
          throw new Error(`missing input '${name}'`);
        }
        input[name] = new onnxruntime.Tensor(t.type, t.data, t.dims);
      });
    }
    const output = await this.binding.run(input);

    const result = new Map<string, onnxjs.Tensor>();
    this.binding.outputNames.forEach((name) => {
      const t = output[name];
      result.set(name, new onnxjs.Tensor(t.data as onnxjs.Tensor.DataType, t.type as onnxjs.Tensor.Type, t.dims));
    });

    return result;
  }

  startProfiling(): void {
    if (this.onnxjsFallback) {
      return this.onnxjsFallback.startProfiling();
    }

    throw new Error('Method not implemented.');
  }

  endProfiling(): void {
    if (this.onnxjsFallback) {
      return this.onnxjsFallback.endProfiling();
    }

    throw new Error('Method not implemented.');
  }
}
