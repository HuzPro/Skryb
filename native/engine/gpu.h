// One source for both GPU vendors: HIP on AMD (gfx1200), CUDA on NVIDIA (RTX 30, 40 and 50 series). The code is
// written against HIP's runtime names; under nvcc they map onto CUDA's, which HIP mirrors one for one.
#pragma once

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;
typedef __half f16;

#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipErrorNotReady cudaErrorNotReady
#define hipGetErrorString cudaGetErrorString
#define hipSetDevice cudaSetDevice
#define hipDeviceGetAttribute cudaDeviceGetAttribute
#define hipDeviceAttributeMultiprocessorCount cudaDevAttrMultiProcessorCount
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipHostMalloc cudaHostAlloc
#define hipHostMallocDefault cudaHostAllocDefault
#define hipHostFree cudaFreeHost
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemsetAsync cudaMemsetAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipStream_t cudaStream_t
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamBeginCapture cudaStreamBeginCapture
#define hipStreamEndCapture cudaStreamEndCapture
#define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal
#define hipGraph_t cudaGraph_t
#define hipGraphExec_t cudaGraphExec_t
#define hipGraphDestroy cudaGraphDestroy
#define hipGraphExecDestroy cudaGraphExecDestroy
#define hipGraphLaunch cudaGraphLaunch
// CUDA 12 dropped the error-node and log arguments.
#define hipGraphInstantiate(exec, graph, node, log, size) cudaGraphInstantiate(exec, graph, 0)
#define hipEvent_t cudaEvent_t
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDisableTiming cudaEventDisableTiming
#define hipEventRecord cudaEventRecord
#define hipEventQuery cudaEventQuery
#define hipEventDestroy cudaEventDestroy
// Every shuffle here runs on a full warp.
#define __shfl_xor(v, m, w) __shfl_xor_sync(0xffffffffu, v, m, w)

__host__ __device__ __forceinline__ unsigned short bf16_bits(bf16 x) { return __bfloat16_as_ushort(x); }
#else
#include <hip/hip_runtime.h>

typedef __bf16 bf16;
typedef _Float16 f16;

__host__ __device__ __forceinline__ unsigned short bf16_bits(bf16 x) { return __builtin_bit_cast(unsigned short, x); }
#endif
