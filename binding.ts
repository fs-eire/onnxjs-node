export  interface Tensor {
    data: Float32Array|Uint8Array|Int8Array|Uint16Array|Int16Array|Int32Array|Uint8Array|Float64Array|Uint32Array;
    dims: number[];
    type: number;
}

export  interface InferenceSession {
    loadModel(modelPath: string): void;

    readonly inputNames: string[];
    readonly outputNames: string[];

    run(inputs: Tensor[]): Tensor[];
}

export interface InferenceSessionConstructor {
    new(): InferenceSession;
}

export const binding = require('./bin/onnxruntime.node') as { InferenceSession: InferenceSessionConstructor };
