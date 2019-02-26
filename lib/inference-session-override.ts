import {InferenceSession, Tensor} from 'onnxjs';
import * as Binding from './binding';

const OnnxjsInferenceSession = InferenceSession;

export class OnnxRuntimeInferenceSession implements InferenceSession {
  private onnxjsFallback?: InferenceSession;

  private binding?: Binding.InferenceSession;

  constructor(config?: InferenceSession.Config) {
    let useOnnxRuntime = !config || typeof config.backendHint !== 'string' || config.backendHint === 'onnxruntime';

    if (useOnnxRuntime) {
      this.binding = new Binding.binding.InferenceSession();
    } else {
      this.onnxjsFallback = new OnnxjsInferenceSession(config);
      console.log('fallback');
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
    if (!this.binding) {
      throw new Error('binding is not assigned');
    }

    this.binding.loadModel(arg0);
  }

  async run(inputFeed: InferenceSession.InputType, options?: InferenceSession.RunOptions):
      Promise<ReadonlyMap<string, Tensor>> {
    if (this.onnxjsFallback) {
      return this.onnxjsFallback.run(inputFeed, options);
    }

    if (!this.binding) {
      throw new Error('session not initialized');
    }

    const input = new Array<Binding.Tensor>(this.binding.inputNames.length);
    let output: Array<Binding.Tensor>;
    if (inputFeed instanceof Map) {
      this.binding.inputNames.forEach((name, i) => {
        const t = inputFeed.get(name);
        if (!t) {
          throw new Error(`missing input '${name}'`);
        }
        input[i] = {data: t.data, dims: t.dims, type: getTensorDataTypeFromString(t.type)};
      });
      output = await this.binding.run(input);
    } else if (Array.isArray(inputFeed)) {
      inputFeed.forEach((t, i) => {
        input[i] = {data: t.data, dims: t.dims, type: getTensorDataTypeFromString(t.type)};
      });
      output = await this.binding.run(input);
    } else {
      this.binding.inputNames.forEach((name, i) => {
        const t = (inputFeed as {readonly [name: string]: any})[name];
        if (!t) {
          throw new Error(`missing input '${name}'`);
        }
        input[i] = {data: t.data, dims: t.dims, type: getTensorDataTypeFromString(t.type)};
      });
      output = await this.binding.run(input);
    }

    const result = new Map<string, Tensor>();
    this.binding.outputNames.forEach((name, i) => {
      const t = output[i];
      result.set(name, new Tensor(t.data as Tensor.DataType, getTensorDataTypeFromEnum(t.type), t.dims));
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

function getTensorDataTypeFromString(type: string): number {
  switch (type) {
    case 'float32':
      return 1;
    case 'int32':
      return 6;
    case 'bool':
      return 9;
    default:
      return -1;
  }
}

function getTensorDataTypeFromEnum(type: number): Tensor.Type {
  switch (type) {
    case 1:
      return 'float32';
    case 6:
      return 'int32';
    case 9:
      return 'bool';
    case 8:
      return 'string';
    default:
      throw new Error(`unsupported data type: ${type}`);
  }
}