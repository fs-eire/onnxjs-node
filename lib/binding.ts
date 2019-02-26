export interface Tensor {
  data: Float32Array|Uint8Array|Int8Array|Uint16Array|Int16Array|Int32Array|Uint8Array|Float64Array|Uint32Array;
  dims: number[];
  type: number;
}

export interface InferenceSession {
  loadModel(modelPath: string): void;

  readonly inputNames: string[];
  readonly outputNames: string[];

  run(inputs: Tensor[]): Tensor[];
}

export interface InferenceSessionConstructor {
  new(): InferenceSession;
}

// construct binding file path
const GPU_ENABLED = false;  // TODO: handle GPU

export const binding =
    require(`../bin/napi-v3/${GPU_ENABLED ? 'gpu' : 'cpu'}/onnxruntime${GPU_ENABLED ? '_gpu' : ''}.node`) as
    {InferenceSession: InferenceSessionConstructor};
