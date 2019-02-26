#ifndef ONNXRUNTIME_NODE_INFERENCE_SESSION_H
#define ONNXRUNTIME_NODE_INFERENCE_SESSION_H

#pragma once

#include <vector>
#include <string>

#include "core/session/onnxruntime_c_api.h"

#include "tensor.h"

class InferenceSession {
 public:
  static void Init();
  InferenceSession();
  ~InferenceSession();

  void LoadModel(const ORTCHAR_T *modelFilePath);

  const std::vector<std::string> & GetInputNames() const { return inputNames_; }
  const std::vector<std::string> & GetOutputNames() const { return outputNames_; }

  std::vector<Tensor> Run(const std::vector<Tensor> &inputs);

 private:
  OrtSessionOptions *sessionOptions_;
  OrtSession *session_;
  OrtAllocatorInfo *allocatorInfo_;

  std::vector<std::string> inputNames_;
  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<ONNXTensorElementDataType> inputDataTypes_;
  std::vector<std::string> outputNames_;
  std::vector<std::vector<int64_t>> outputShapes_;
  std::vector<ONNXTensorElementDataType> outputDataTypes_;
};

#endif
