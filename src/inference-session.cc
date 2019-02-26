#include <memory>
#include <stdexcept>
#include <sstream>
#include <core/providers/cpu/cpu_provider_factory.h>

#include "inference-session.h"

static OrtEnv * g_env;

void InferenceSession::Init() {
  // Create Env
  auto status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxjs", &g_env);
  if (status) {
    std::ostringstream what;
    what << "Failed to create onnxruntime environment: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }
}

InferenceSession::InferenceSession() {
  // Create allocation info
  auto status = OrtCreateCpuAllocatorInfo(OrtDeviceAllocator, OrtMemTypeDefault, &this->allocatorInfo_);
  if (status) {
    std::ostringstream what;
    what << "Failed to create allocation info: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
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
}

void InferenceSession::LoadModel(const ORTCHAR_T *modelPath) {
  // Create session and load model
  auto status = OrtCreateSession(g_env, modelPath, this->sessionOptions_, &this->session_);
  if (status) {
    std::ostringstream what;
    what << "Failed to load model: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }

  // Initialize metadata

  // STEP.1 - Create a temporary allocator
  OrtAllocator *defaultAllocator;
  status = OrtCreateDefaultAllocator(&defaultAllocator);
  if (status) {
    std::ostringstream what;
    what << "Failed to create default allocator: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }
  std::unique_ptr<OrtAllocator, decltype(&OrtReleaseAllocator)> allocator(defaultAllocator, OrtReleaseAllocator);

  // STEP.2 - Get input/output count
  size_t inputCount;
  status = OrtSessionGetInputCount(this->session_, &inputCount);
  if (status) {
    std::ostringstream what;
    what << "Failed to get model input count: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }
  size_t outputCount;
  status = OrtSessionGetOutputCount(this->session_, &outputCount);
  if (status) {
    std::ostringstream what;
    what << "Failed to get model output count: " << OrtGetErrorMessage(status);
    throw std::runtime_error(what.str());
  }

  // STEP.3 - Get input/output names
  this->inputNames_.reserve(inputCount);
  for (size_t i = 0; i < inputCount; i++) {
    char *name;
    status = OrtSessionGetInputName(this->session_, i, allocator.get(), &name);
    if (status) {
      std::ostringstream what;
      what << "Failed to get model input name: " << OrtGetErrorMessage(status);
      throw std::runtime_error(what.str());
    }
    this->inputNames_.emplace_back(name);
    OrtAllocatorFree(allocator.get(), name);
  }
  this->outputNames_.reserve(outputCount);
  for (size_t i = 0; i < outputCount; i++) {
    char *name;
    status = OrtSessionGetOutputName(this->session_, i, allocator.get(), &name);
    if (status) {
      std::ostringstream what;
      what << "Failed to get model output name: " << OrtGetErrorMessage(status);
      throw std::runtime_error(what.str());
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
      std::ostringstream what;
      what << "Failed to get input type info: " << OrtGetErrorMessage(status);
      throw std::runtime_error(what.str());
    }
    const OrtTensorTypeAndShapeInfo* tensorInfo = OrtCastTypeInfoToTensorInfo(typeInfo);

    // Element type
    auto dataType = OrtGetTensorElementType(tensorInfo);
    this->inputDataTypes_.push_back(dataType);

    // Shape
    auto dimsCount = OrtGetNumOfDimensions(tensorInfo);
    std::vector<int64_t> dims(dimsCount);
    if (dimsCount > 0) {
      OrtGetDimensions(tensorInfo, &dims[0], dimsCount);
    }
    this->inputShapes_.push_back(dims);

    OrtReleaseTypeInfo(typeInfo);
  }

  this->outputDataTypes_.reserve(outputCount);
  this->outputShapes_.reserve(outputCount);
  for (size_t i = 0; i < outputCount; i++) {
    OrtTypeInfo *typeInfo;
    status = OrtSessionGetOutputTypeInfo(this->session_, i, &typeInfo);
    if (status) {
      std::ostringstream what;
      what << "Failed to get output type info: " << OrtGetErrorMessage(status);
      throw std::runtime_error(what.str());
    }
    const OrtTensorTypeAndShapeInfo* tensorInfo = OrtCastTypeInfoToTensorInfo(typeInfo);

    // Element type
    auto dataType = OrtGetTensorElementType(tensorInfo);
    this->outputDataTypes_.push_back(dataType);

    // Shape
    auto dimsCount = OrtGetNumOfDimensions(tensorInfo);
    std::vector<int64_t> dims(dimsCount);
    if (dimsCount > 0) {
      OrtGetDimensions(tensorInfo, &dims[0], dimsCount);
    }
    this->outputShapes_.push_back(dims);

    OrtReleaseTypeInfo(typeInfo);
  }
}

std::vector<Tensor> InferenceSession::Run(const std::vector<Tensor> &inputTensors) {
  size_t outputCount = this->GetOutputNames().size();
  if (outputCount == 0) {
    return std::vector<Tensor>();
  }

  size_t inputCount = inputTensors.size();
  std::vector<const char *> inputNames(inputCount);
  std::vector<OrtValue *> inputs(inputCount);
  for (size_t i = 0 ; i < inputCount; i++) {
    inputNames[i] = inputTensors[i].name;
    auto status = OrtCreateTensorWithDataAsOrtValue(this->allocatorInfo_,
                                                    inputTensors[i].data,
                                                    inputTensors[i].dataLength,
                                                    inputTensors[i].shape.empty() ? nullptr : &inputTensors[i].shape[0],
                                                    inputTensors[i].shape.size(),
                                                    inputTensors[i].type, &inputs[i]);
    if (status) {
      std::ostringstream what;
      what << "Failed to create input tensors: " << OrtGetErrorMessage(status);
      throw std::runtime_error(what.str());
    }
  }

  std::vector<const char *> outputNames(outputCount);
  for (size_t i = 0 ; i < outputCount; i++) {
    outputNames[i] = this->GetOutputNames()[i].c_str();
  }

  std::vector<OrtValue *> outputs(outputCount);

  auto status = OrtRun(this->session_,
                       nullptr,
                       inputNames.empty() ? nullptr : &inputNames[0],
                       inputs.empty() ? nullptr : &inputs[0],
                       inputNames.size(),
                       &outputNames[0],
                       outputNames.size(),
                       &outputs[0]);
  if (status) {
      std::ostringstream what;
      what << "Failed to run the model: " << OrtGetErrorMessage(status);
      throw std::runtime_error(what.str());
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
    auto &tensor = Tensor::From(value);
    outputTensors.push_back(tensor);
  }
  
  return std::move(outputTensors);
}
