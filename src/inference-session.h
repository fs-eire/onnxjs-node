#ifndef ONNXRUNTIME_NODE_INFERENCE_SESSION_H
#define ONNXRUNTIME_NODE_INFERENCE_SESSION_H

#pragma once

#include <vector>
#include <string>

#include "core/session/onnxruntime_c_api.h"

#include "tensor.h"

class InferenceSession {
 public:
  InferenceSession();
  ~InferenceSession();

  void LoadModel(const ORTCHAR_T *modelFilePath);

  const std::vector<std::string> & GetInputNames() const { return inputNames_; }
  const std::vector<std::string> & GetOutputNames() const { return outputNames_; }
  ONNXTensorElementDataType GetInputDataType(uint32_t index) const { return inputDataTypes_[index]; }
  const OrtAllocatorInfo * GetAllocationInfo() const { return allocatorInfo_; }

  std::vector<Tensor> Run(std::vector<Tensor> inputs);

 private:
  OrtEnv *env_;
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
