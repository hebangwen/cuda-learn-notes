#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// FP16/BF16 CUDA Cores/Tensor Cores: 
// https://resources.nvidia.com/en-us-tensor-core 
// Non MatMul FP16/BF16 -> CUDA Cores
//     MatMul FP16/BF16 -> Tensor Cores
//       Non MatMul FP8 -> Not supported
//           MatMul FP8 -> Tensor Cores

// CUDA温故(0x00): 一步步学习block all reduce: 从FP32到FP16/BF16，再到FP8
// -------------------------------------- FP32 -------------------------------------- 
// Warp Reduce Sum

// 线程束洗牌函数： http://www.zh0ngtian.tech/posts/ada27037.html
// shfl -> shuffle，而不是 shift left

// 这个函数把单个 warp 内的数据并行求和到 val 中
// 传入的参数实际上就是整个 warp 里的所有寄存器 val
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  // 函数作用，让当前线程束能够访问同一个 warp 内部的寄存器而不需要经过 smem
  // 所以能够实现一个 warp 内部归约算法
  // 同时，异或相当于一个加法，当第一次 mask=16 时，thread-0 得到 thread-16 寄存器的值，这样前 16 个元素就得到了 32 个元素之和
  // 下一步，当 mask=8，thread-0 得到 thread-8 的寄存器，前 8 个元素保存了 32 个元素之和
  // 一直迭代直到 mask=0
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * NUM_THREADS;
    constexpr int WARP_NUM = NUM_THREADS / WARP_SIZE;
    __shared__ float smem[WARP_NUM];
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float val = gid < N ? a[gid] : 0.0f;
    // 256 -> 256/32 = 8
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    // 8 -> 1
    val = lane < WARP_NUM ? smem[lane] : 0.0f;
    if (warp == 0) val = warp_reduce_sum_f32<WARP_NUM>(val);
    // N/256 -> 1
    if (tid == 0) atomicAdd(y, val);
    // [END MANUAL IMPLEMENTATION]
}

// Block All Reduce Sum + float4
// grid(N/256), block(256/4)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256/4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

// -------------------------------------- FP16 -------------------------------------- 
// Warp Reduce Sum: Half
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    // val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
  float val_f32 = __half2float(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// Block All Reduce Sum: Half
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

// -------------------------------------- BF16 -------------------------------------- 
// Warp Reduce Sum: Half
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16_bf16(
  __nv_bfloat16 val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(
  __nv_bfloat16 val) {
  float val_f32 = __bfloat162float(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
  }
  return val_f32;
}

// Block All Reduce Sum: BF16
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(
  __nv_bfloat16* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(
  __nv_bfloat16* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

// -------------------------------------- FP8 -------------------------------------- 
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e4m3_f16(
  __nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_fp8_e5m2_f16(
  __nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
  half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
  }
  return val_f16;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/16>
__global__ void block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/16>
__global__ void block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(
  __nv_fp8_storage_t* a, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

// -------------------------------------- INT8 -------------------------------------- 
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
  int32_t val_i32 = static_cast<int32_t>(val);
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
  }
  return val_i32;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(
  int8_t* a, int32_t* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(
  int8_t* a, int32_t* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define LANUCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type, out_type)   \
block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)><<<grid, block>>>(  \
  reinterpret_cast<element_type*>(x.data_ptr()),                                  \
  reinterpret_cast<out_type*>(y.data_ptr()), N);  

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type, n_elements, out_type) \
  const int NT = (K)/(n_elements);                                                           \
  dim3 block(NT);                                                                            \
  dim3 grid((S));                                                                            \
  switch (NT)                                                                                \
  {                                                                                          \
  case 32:                                                                                   \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)                  \
    break;                                                                                   \
  case 64:                                                                                   \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)                  \
    break;                                                                                   \
  case 128:                                                                                  \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)                 \
    break;                                                                                   \
  case 256:                                                                                  \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)                 \
    break;                                                                                   \
  case 512:                                                                                  \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)                 \
    break;                                                                                   \
  case 1024:                                                                                 \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)                \
    break;                                                                                   \
  default:                                                                                   \
    throw std::runtime_error(                                                                \
      "only support (K)/(n_elements): 32/64/128/256/512/1024");                              \
    break;                                                                                   \
  } 

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type, n_elements, out_type) \
torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(torch::Tensor x) {                 \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                         \
  auto y_th_type = (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;                  \
  auto options = torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);                \
  auto y = torch::zeros({1}, options);                                                           \
  const int ndim = x.dim();                                                                      \
  if (ndim != 2) {                                                                               \
    int N = 1;                                                                                   \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                                           \
    dim3 block(1024 / (n_elements));                                                             \
    dim3 grid((N + 1024 - 1) / 1024);                                                            \
    block_all_reduce_sum_##packed_type##_##acc_type##_kernel<                                    \
      1024 / (n_elements)><<<grid, block>>>(                                                     \
      reinterpret_cast<element_type*>(x.data_ptr()),                                             \
      reinterpret_cast<out_type*>(y.data_ptr()), N);                                             \
  } else {                                                                                       \
    const int S = x.size(0);                                                                     \
    const int K = x.size(1);                                                                     \
    const int N = S * K;                                                                         \
    if ((K/(n_elements)) <= 1024) {                                                              \
      DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type, n_elements, out_type)       \
    } else {                                                                                     \
      int N = 1;                                                                                 \
      for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                                         \
      dim3 block(1024 / (n_elements));                                                           \
      dim3 grid((N + 1024 - 1) / 1024);                                                          \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<                                  \
        1024 / (n_elements)><<<grid, block>>>(                                                   \
        reinterpret_cast<element_type*>(x.data_ptr()),                                           \
        reinterpret_cast<out_type*>(y.data_ptr()), N);                                           \
    }                                                                                            \
  }                                                                                              \
  return y;                                                                                      \
}

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32,              f32,  torch::kFloat32,       float,              1,  float)
TORCH_BINDING_REDUCE(f32x4,            f32,  torch::kFloat32,       float,              4,  float)
TORCH_BINDING_REDUCE(f16,              f16,  torch::kHalf,          half,               1,  float)
TORCH_BINDING_REDUCE(f16,              f32,  torch::kHalf,          half,               1,  float)
TORCH_BINDING_REDUCE(f16x2,            f16,  torch::kHalf,          half,               2,  float)
TORCH_BINDING_REDUCE(f16x2,            f32,  torch::kHalf,          half,               2,  float)
TORCH_BINDING_REDUCE(f16x8_pack,       f16,  torch::kHalf,          half,               8,  float)
TORCH_BINDING_REDUCE(f16x8_pack,       f32,  torch::kHalf,          half,               8,  float)
TORCH_BINDING_REDUCE(bf16,             bf16, torch::kBFloat16,      __nv_bfloat16,      1,  float)
TORCH_BINDING_REDUCE(bf16,             f32,  torch::kBFloat16,      __nv_bfloat16,      1,  float)
TORCH_BINDING_REDUCE(bf16x2,           bf16, torch::kBFloat16,      __nv_bfloat16,      2,  float)
TORCH_BINDING_REDUCE(bf16x2,           f32,  torch::kBFloat16,      __nv_bfloat16,      2,  float)
TORCH_BINDING_REDUCE(bf16x8_pack,      bf16, torch::kBFloat16,      __nv_bfloat16,      8,  float)
TORCH_BINDING_REDUCE(bf16x8_pack,      f32,  torch::kBFloat16,      __nv_bfloat16,      8,  float)
TORCH_BINDING_REDUCE(fp8_e4m3,         f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 1,  float)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16,  torch::kFloat8_e4m3fn, __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2,         f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 1,  float)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16,  torch::kFloat8_e5m2,   __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(i8,               i32,  torch::kInt8,          int8_t,             1,  int32_t)
TORCH_BINDING_REDUCE(i8x16_pack,       i32,  torch::kInt8,          int8_t,             16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}
