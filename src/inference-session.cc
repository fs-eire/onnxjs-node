#include <memory>
#include <stdexcept>
#include <core/providers/cpu/cpu_provider_factory.h>

#include "inference-session.h"

InferenceSession::InferenceSession() {
  // Create Env
  auto status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxjs", &this->env_);
  if (status) {
    throw std::runtime_error("Failed to create onnxruntime environment");
  }

  // Create allocation info
  status = OrtCreateCpuAllocatorInfo(OrtDeviceAllocator, OrtMemTypeDefault, &this->allocatorInfo_);
  if (status) {
    throw std::runtime_error("Failed to create allocation info");
  }

  // Create Session Options
  this->sessionOptions_ = OrtCreateSessionOptions();

  this->session_ = nullptr;
}

InferenceSession::~InferenceSession() {
  if (this->session_) {
    OrtReleaseSession(this->session_);
    this->session_ = nullptr;
  }

  OrtReleaseSessionOptions(this->sessionOptions_);
  this->sessionOptions_ = nullptr;

  OrtReleaseAllocatorInfo(this->allocatorInfo_);
  this->allocatorInfo_ = nullptr;

  OrtReleaseEnv(this->env_);
  this->env_ = nullptr;
}

void InferenceSession::LoadModel(const ORTCHAR_T *modelPath) {
  // Create session and load model
  auto status = OrtCreateSession(this->env_, modelPath, this->sessionOptions_, &this->session_);
  if (status) {
    throw std::runtime_error("Failed to load model");
  }

  // Initialize metadata

  // STEP.1 - Create a temporary allocator
  OrtAllocator *defaultAllocator;
  status = OrtCreateDefaultAllocator(&defaultAllocator);
  if (status) {
    throw std::runtime_error("Failed to create default allocator");
  }
  std::unique_ptr<OrtAllocator, decltype(&OrtReleaseAllocator)> allocator(defaultAllocator, OrtReleaseAllocator);

  // STEP.2 - Get input/output count
  size_t inputCount;
  status = OrtSessionGetInputCount(this->session_, &inputCount);
  if (status) {
    throw std::runtime_error("Failed to get model input count");
  }
  size_t outputCount;
  status = OrtSessionGetOutputCount(this->session_, &outputCount);
  if (status) {
    throw std::runtime_error("Failed to get model output count");
  }

  // STEP.3 - Get input/output names
  this->inputNames_.reserve(inputCount);
  for (size_t i = 0; i < inputCount; i++) {
    char *name;
    status = OrtSessionGetInputName(this->session_, i, allocator.get(), &name);
    if (status) {
      throw std::runtime_error("Failed to get model input name");
    }
    this->inputNames_.emplace_back(name);
    OrtAllocatorFree(allocator.get(), name);
  }
  this->outputNames_.reserve(outputCount);
  for (size_t i = 0; i < outputCount; i++) {
    char *name;
    status = OrtSessionGetOutputName(this->session_, i, allocator.get(), &name);
    if (status) {
      throw std::runtime_error("Failed to get model output name");
    }
    this->outputNames_.emplace_back(name);
    OrtAllocatorFree(allocator.get(), name);
  }

  // STEP.4 - Get input/output type info
  this->inputDataTypes_.reserve(inputCount);
  this->inputShapes_.reserve(inputCount);
  for (size_t i = 0; i < inputCount; i++) {
    OrtTypeInfo *typeInfo;
    status = OrtSessionGetInputTypeInfo(this->session_, i, &typeInfo);
    if (status) {
      throw std::runtime_error("Failed to get input type info");
    }
    const OrtTensorTypeAndShapeInfo* tensorInfo = OrtCastTypeInfoToTensorInfo(typeInfo);

    // Element type
    auto dataType = OrtGetTensorElementType(tensorInfo);
    this->inputDataTypes_.push_back(dataType);

    // Shape
    auto dimsCount = OrtGetNumOfDimensions(tensorInfo);
    std::vector<int64_t> dims(dimsCount);
    OrtGetDimensions(tensorInfo, &dims[0], dimsCount);
    this->inputShapes_.push_back(dims);

    OrtReleaseTypeInfo(typeInfo);
  }

  this->outputDataTypes_.reserve(outputCount);
  this->outputShapes_.reserve(outputCount);
  for (size_t i = 0; i < outputCount; i++) {
    OrtTypeInfo *typeInfo;
    status = OrtSessionGetOutputTypeInfo(this->session_, i, &typeInfo);
    if (status) {
      throw std::runtime_error("Failed to get output type info");
    }
    const OrtTensorTypeAndShapeInfo* tensorInfo = OrtCastTypeInfoToTensorInfo(typeInfo);

    // Element type
    auto dataType = OrtGetTensorElementType(tensorInfo);
    this->outputDataTypes_.push_back(dataType);

    // Shape
    auto dimsCount = OrtGetNumOfDimensions(tensorInfo);
    std::vector<int64_t> dims(dimsCount);
    OrtGetDimensions(tensorInfo, &dims[0], dimsCount);
    this->outputShapes_.push_back(dims);

    OrtReleaseTypeInfo(typeInfo);
  }
}

std::vector<Tensor> InferenceSession::Run(std::vector<Tensor> inputTensors) {
  size_t inputCount = inputTensors.size();
  std::vector<const char *> inputNames(inputCount);
  std::vector<OrtValue *> inputs(inputCount);
  for (size_t i = 0 ; i < inputCount; i++) {
    inputNames[i] = inputTensors[i].name;
    auto status = OrtCreateTensorWithDataAsOrtValue(this->allocatorInfo_,
                                                    inputTensors[i].data,
                                                    inputTensors[i].dataLength,
                                                    &inputTensors[i].shape[0],
                                                    inputTensors[i].shape.size(),
                                                    inputTensors[i].type, &inputs[i]);
    if (status) {
      throw std::runtime_error("Failed to create input tensors");
    }
  }

  size_t outputCount = this->GetOutputNames().size();
  std::vector<const char *> outputNames(outputCount);
  for (size_t i = 0 ; i < outputCount; i++) {
    outputNames[i] = this->GetOutputNames()[i].c_str();
  }

  std::vector<OrtValue *> outputs(outputCount);

  auto status = OrtRun(this->session_, nullptr, &inputNames[0], &inputs[0], inputNames.size(), &outputNames[0], outputNames.size(), &outputs[0]);
  if (status) {
      throw std::runtime_error("Failed to run the model");
  }

  // Release input values
  for (size_t i = 0 ; i < inputCount; i++) {
    OrtReleaseValue(inputs[i]);
  }

  // Feed output tensors
  std::vector<Tensor> outputTensors;
  outputTensors.reserve(outputCount);
  for (size_t i = 0 ; i < outputCount; i++) {
    auto value = outputs[i];
    if (ONNX_TYPE_TENSOR != OrtGetValueType(value)) {
      throw std::runtime_error("Unsupported value type. Only Tensor value is supported.");
    }

    void *data;
    status = OrtGetTensorMutableData(value, &data);
    if (status) {
      throw std::runtime_error("Failed to get data from output tensors");
    }

    OrtTensorTypeAndShapeInfo *tensorInfo;
    status = OrtGetTensorShapeAndType(value, &tensorInfo);
    if (status) {
      throw std::runtime_error("Failed to get tensor info from output tensors");
    }

    auto dataType = OrtGetTensorElementType(tensorInfo);
    auto dimsCount = OrtGetNumOfDimensions(tensorInfo);
    std::vector<int64_t> dims(dimsCount);
    OrtGetDimensions(tensorInfo, &dims[0], dimsCount);
    auto elementCount = OrtGetTensorShapeElementCount(tensorInfo);
    if (elementCount <= 0) {
      throw std::runtime_error("Invalid element count");
    }

    if (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT != dataType) {
      throw std::runtime_error("currently only support FLOAT32 element type");
    }

    std::vector<size_t> dimsUnsigned(dimsCount);
    for (size_t i = 0 ; i < dimsCount; i++) {
      dimsUnsigned[i] = static_cast<size_t>(dims[i]);
    }

    Tensor tensor{data, static_cast<size_t>(elementCount * 4), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dimsUnsigned};
    outputTensors.push_back(tensor);

    OrtReleaseTensorTypeAndShapeInfo(tensorInfo);
  }
  
  return outputTensors;
}
