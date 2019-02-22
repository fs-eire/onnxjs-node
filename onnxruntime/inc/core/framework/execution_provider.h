// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/framework/func_api.h"

namespace onnxruntime {
class GraphViewer;
class Node;
}  // namespace onnxruntime
namespace onnxruntime {

struct ComputeCapability;
class KernelRegistry;
class KernelRegistryManager;

/**
   Logical device representation.
*/
typedef std::map<int, AllocatorPtr> AllocatorMap;

// if we are export the fused function to dll, the function will still in the same binary as lotus
// use std function to give execution provider some chance to capture some state.
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<int(FunctionState, ONNXRunTimeTensor*, size_t, ONNXRunTimeTensor*, size_t)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};

class IExecutionProvider {
 public:
  virtual ~IExecutionProvider() = default;

  /**
     Get all IAllocators for <*this> execution provider.
  */
  std::vector<AllocatorPtr> GetAllocatorMap() const {
    std::vector<AllocatorPtr> values;
    for (auto& kv : allocators_) {
      values.push_back(kv.second);
    }
    return values;
  }

  /**
     Get allocator with specified MemType
  */
  virtual AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const;

  /**
     Get execution provider's capability for the specified <graph>.
     Return a bunch of IndexedSubGraphs <*this> execution provider can run if
     the sub-graph contains only one node or can fuse to run if the sub-graph
     contains more than one node. The node indexes contained in sub-graphs may
     have overlap, and it's ONNXRuntime's responsibility to do the partition
     and decide whether a node will be assigned to <*this> execution provider.
  */
  virtual std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const;

  /**
     Get kernel registry per execution provider type.
     The KernelRegistry share pointer returned is shared across sessions.

     NOTE: this is a tricky but final solution to achieve following goals,
     1. The execution provider type based kernel registry should be shared
     across sessions.
     Only one copy of this kind of kernel registry exists in ONNXRuntime
     with multiple sessions/models.
     2. Adding an execution provider into ONNXRuntime does not need to touch ONNXRuntime
     frameowrk/session code.
     3. onnxruntime (framework/session) does not depend on any specific
     execution provider lib.
  */
  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const = 0;

  /**
     Copy tensor between execution providers
  */
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst) const = 0;

  /**
     Copy tensor between execution providers on specified exec queue
  */
  virtual common::Status CopyTensor(const Tensor& src, Tensor& dst,
                                    int exec_queue_id) const;

  /**
     Returns an opaque handle whose exact type varies based on the provider
     and is interpreted accordingly by the corresponding kernel implementation.
     For Direct3D operator kernels, this may return an IUnknown supporting
     QueryInterface to ID3D12GraphicsCommandList1.
  */
  virtual const void* GetExecutionHandle() const noexcept = 0;

  /**
     @return type of the execution provider; should match that set in the node
     through the SetExecutionProvider API. Example valid return values are:
     kCpuExecutionProvider, kCudaExecutionProvider
  */
  virtual std::string Type() const = 0;

  /**
     Blocks until the device has completed all preceding requested tasks.
     Currently this is primarily used by the IOBinding object to ensure that all
     inputs have been copied to the device before execution begins.
  */
  virtual common::Status Sync() const;

  /**
     Called when InferenceSession::Run started
     NOTE that due to async execution in provider, the actual work of previous
     Run may not be finished on device This function should be regarded as the
     point after which a new Run would start to submit commands from CPU
  */
  virtual common::Status OnRunStart();

  /**
     Called when InferenceSession::Run ended
     NOTE that due to async execution in provider, the actual work of this Run
     may not be finished on device This function should be regarded as the point
     that all commands of current Run has been submmited by CPU
  */
  virtual common::Status OnRunEnd();

  void InsertAllocator(AllocatorPtr allocator);

  /**
  Given a list of fused_node, return create_state/compute/release_state func for each node.
  */
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_node,
                                 std::vector<NodeComputeInfo>& node_compute_funcs);

  /**
  Given a list of fused_node, return a dll that expose functions for each node.
  For each node, there should be three symbols:
     Create_State_${node_name}
     Compute_${node_name}
     Release_State_${node_name}
  */
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_node,
                                 std::string& dll_path);

 private:
  AllocatorMap allocators_;
};
}  // namespace onnxruntime
