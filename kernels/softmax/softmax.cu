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

// -------------------------------------- FP32 -------------------------------------- 
// DS required for Online Softmax
struct __align__(8) MD { float m; float d; }; 
// Warp Reduce for Online Softmax
template<const int kWarpSize = WARP_SIZE >
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
  #pragma unroll
  for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;
    
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);  
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template<const int NUM_THREADS=256>
__device__ float block_reduce_max_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Softmax x: N, y: N
// grid(N/256), block(K=256)
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_kernel(float* x, float* y, float* total, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = gid < N ? x[gid] : 0.0f;
    // NOTE: atomicMax 只能用于 i32 / ui32 / ll64 / ull64 上，所以这个方式不能计算 safe-softmax

    // float block_max = block_reduce_max_f32<NUM_THREADS>(val);
    // if (tid == 0) atomicMax(total, block_max);
    // __threadfence();

    // // 计算 exp(x - xmax)
    // float exp_val = __expf(val - *total);
    // float block_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    // __threadfence();
    // if (gid == 0) *total = 0.0f;
    // __threadfence();
    // if (tid == 0) atomicAdd(total, block_sum);
    // __threadfence();

    val = __expf(val);
    float block_sum = block_reduce_sum_f32<NUM_THREADS>(val);
    if (tid == 0) atomicAdd(total, block_sum);
    __threadfence();

    val = val / *total;
    if (gid < N) y[gid] = val;
    // [END MANUAL IMPLEMENTATION]
}

// Softmax Vec4 x: N, y: N
// grid(N/256), block(256/4)
template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4_kernel(float* x, float* y, float* total, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    float4 val = gid < N ? FLOAT4(x[gid]) : float4{0.0f, 0.0f, 0.0f, 0.0f};
    val.x = __expf(val.x);
    val.y = __expf(val.y);
    val.z = __expf(val.z);
    val.w = __expf(val.w);
    float sum = val.x + val.y + val.z + val.w;
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    if (tid == 0) atomicAdd(total, sum);
    __threadfence();

    sum = 1.0f / *total;
    val.x = val.x * sum;
    val.y = val.y * sum;
    val.z = val.z * sum;
    val.w = val.w * sum;

    if (gid < N) FLOAT4(y[gid]) = val;
    // [END MANUAL IMPLEMENTATION]
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = gid < N ? x[gid] : 0.0f;
    val = __expf(val);
    float sum = block_reduce_sum_f32<NUM_THREADS>(val);
    val = val / sum;
    if (gid < N) y[gid] = val;
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    float4 val = gid < N ? FLOAT4(x[gid]) : float4{0.0f, 0.0f, 0.0f, 0.0f};
    val.x = __expf(val.x);
    val.y = __expf(val.y);
    val.z = __expf(val.z);
    val.w = __expf(val.w);
    float sum = val.x + val.y + val.z + val.w;
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    sum = 1.0f / sum;
    val.x = val.x * sum;
    val.y = val.y * sum;
    val.z = val.z * sum;
    val.w = val.w * sum;
    if (gid < N) FLOAT4(y[gid]) = val;
    // [END MANUAL IMPLEMENTATION]
}

// safe_softmax per token
template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = gid < N ? x[gid] : 0.0f;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val);
    val -= max_val;
    val = __expf(val);
    float sum = block_reduce_sum_f32<NUM_THREADS>(val);
    val = val / sum;
    if (gid < N) y[gid] = val;
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256/4>
__global__ void safe_softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    float4 val = gid < N ? FLOAT4(x[gid]) : float4{0.0f, 0.0f, 0.0f, 0.0f};
    float max_val = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
    max_val = block_reduce_max_f32<NUM_THREADS>(max_val);
    val.x = __expf(val.x - max_val);
    val.y = __expf(val.y - max_val);
    val.z = __expf(val.z - max_val);
    val.w = __expf(val.w - max_val);
    float sum = val.x + val.y + val.z + val.w;
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    sum = 1.0f / sum;
    val.x = val.x * sum;
    val.y = val.y * sum;
    val.z = val.z * sum;
    val.w = val.w * sum;
    if (gid < N) FLOAT4(y[gid]) = val;
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = gid < N ? __half2float(x[gid]) : 0.0f;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val);
    val -= max_val;
    val = __expf(val);
    float sum = block_reduce_sum_f32<NUM_THREADS>(val);
    val = val / sum;
    if (gid < N) y[gid] = __float2half(val);
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    half2 val = gid < N ? HALF2(x[gid]) : half2{__float2half(0.0f), __float2half(0.0f)};
    float2 val_f32 = {__half2float(val.x), __half2float(val.y)};
    float max_val = fmaxf(val_f32.x, val_f32.y);
    max_val = block_reduce_max_f32<NUM_THREADS>(max_val);
    val_f32.x = __expf(val_f32.x - max_val);
    val_f32.y = __expf(val_f32.y - max_val);
    float sum = val_f32.x + val_f32.y;
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    val_f32.x = val_f32.x / sum;
    val_f32.y = val_f32.y / sum;
    val = half2{__float2half(val_f32.x), __float2half(val_f32.y)};
    if (gid < N) HALF2(y[gid]) = val;
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x;
    int gid = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    half val[8];
    float val_f32[8];
    LDST128BITS(val[0]) = LDST128BITS(x[gid]);
    float max_val = __half2float(val[0]);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      val_f32[i] = __half2float(val[i]);
      max_val = fmaxf(val_f32[i], max_val);
    }
    max_val = block_reduce_max_f32<NUM_THREADS>(max_val);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      val_f32[i] = __expf(val_f32[i] - max_val);
      sum += val_f32[i];
    }
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    sum = 1.0f / sum;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      val[i] = __float2half(val_f32[i] * sum);
    }
    if (gid * 8 < N) LDST128BITS(y[gid]) = LDST128BITS(val[0]);
    // [END MANUAL IMPLEMENTATION]
}

template<const int NUM_THREADS = 256 >
__global__ void online_safe_softmax_f32_per_token_kernel(const float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

template <const int NUM_THREADS = 256 / 4>
__global__ void online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N)
{
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

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)               \
assert((T1).dim() == (T2).dim());                      \
for (int i = 0; i < (T1).dim(); ++i) {                 \
  if ((T2).size(i) != (T1).size(i)) {                  \
    throw std::runtime_error("Tensor size mismatch!"); \
  }                                                    \
}

// grid memory fence
#define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)    \
void softmax_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                         \
  auto options = torch::TensorOptions().dtype((th_type)).device(torch::kCUDA, 0);\
  const int N = x.size(0);                                                       \
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                 \
  auto total = torch::zeros({1}, options);                                       \
  dim3 block(256);                                                               \
  dim3 grid(((N + 256 - 1) / 256) / (n_elements));                               \
  softmax_##packed_type##_kernel<256><<<grid, block>>>(                          \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()),                             \
      reinterpret_cast<element_type*>(total.data_ptr()), N);                     \
}

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)       \
softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),        \
      reinterpret_cast<float*>(y.data_ptr()),        \
      N);  

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                  \
  dim3 grid((S));                                   \
  switch ((H))                                      \
  {                                                 \
  case 32:                                          \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)         \
    break;                                          \
  case 64:                                          \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)         \
    break;                                          \
  case 128:                                         \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)        \
    break;                                          \
  case 256:                                         \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)        \
    break;                                          \
  case 512:                                         \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)        \
    break;                                          \
  case 1024:                                        \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                          \
  default:                                          \
    throw std::runtime_error(                       \
      "only support H: 64/128/256/512/1024");       \
    break;                                          \
  } 

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)  \
softmax_f32x4_per_token_kernel<(H)/4><<<          \
      grid, block>>>(                             \
      reinterpret_cast<float*>(x.data_ptr()),     \
      reinterpret_cast<float*>(y.data_ptr()),     \
      N);  

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/4;                               \
  dim3 block(NT);                                     \
  dim3 grid((S));                                     \
  switch (H)                                          \
  {                                                   \
  case 32:                                            \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32)         \
    break;                                            \
  case 64:                                            \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64)         \
    break;                                            \
  case 128:                                           \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128)        \
    break;                                            \
  case 256:                                           \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256)        \
    break;                                            \
  case 512:                                           \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512)        \
    break;                                            \
  case 1024:                                          \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024)       \
    break;                                            \
  case 2048:                                          \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048)       \
    break;                                            \
  case 4096:                                          \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096)       \
    break;                                            \
  default:                                            \
    throw std::runtime_error(                         \
      "only support H: 64/128/.../1024*4");           \
    break;                                            \
  } 

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)       \
safe_softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),             \
      reinterpret_cast<float*>(y.data_ptr()),             \
      N);  

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                       \
  dim3 grid((S));                                        \
  switch ((H))                                           \
  {                                                      \
  case 32:                                               \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)         \
    break;                                               \
  case 64:                                               \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)         \
    break;                                               \
  case 128:                                              \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)        \
    break;                                               \
  case 256:                                              \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)        \
    break;                                               \
  case 512:                                              \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)        \
    break;                                               \
  case 1024:                                             \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                               \
  default:                                               \
    throw std::runtime_error(                            \
      "only support H: 64/128/256/512/1024");            \
    break;                                               \
  } 

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)            \
online_safe_softmax_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),                    \
      reinterpret_cast<float*>(y.data_ptr()),                    \
      N);  

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                       \
  dim3 grid((S));                                        \
  switch ((H))                                           \
  {                                                      \
  case 32:                                               \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)       \
    break;                                               \
  case 64:                                               \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)       \
    break;                                               \
  case 128:                                              \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)      \
    break;                                               \
  case 256:                                              \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)      \
    break;                                               \
  case 512:                                              \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)      \
    break;                                               \
  case 1024:                                             \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)     \
    break;                                               \
  default:                                               \
    throw std::runtime_error(                            \
      "only support H: 64/128/256/512/1024");            \
    break;                                               \
  } 

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)              \
online_safe_softmax_f32x4_pack_per_token_kernel<(H/4)><<<grid, block>>>(  \
      reinterpret_cast<float*>(x.data_ptr()),                             \
      reinterpret_cast<float*>(y.data_ptr()),                             \
      N);  

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)                     \
  dim3 block((H/4));                                                                  \
  dim3 grid((S));                                                                     \
  switch ((H))                                                                        \
  {                                                                                   \
  case 128:                                                                           \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)                            \
    break;                                                                            \
  case 256:                                                                           \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)                            \
    break;                                                                            \
  case 512:                                                                           \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)                            \
    break;                                                                            \
  case 1024:                                                                          \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)                           \
    break;                                                                            \
  case 2048:                                                                          \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)                           \
    break;                                                                            \
  case 4096:                                                                          \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)                           \
    break;                                                                            \
  default:                                                                            \
    throw std::runtime_error(                                                         \
      "only support H: 128/256/.../4096;");                                           \
    break;                                                                            \
  } 

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)   \
safe_softmax_f32x4_per_token_kernel<(H)/4><<<           \
      grid, block>>>(                                   \
      reinterpret_cast<float*>(x.data_ptr()),           \
      reinterpret_cast<float*>(y.data_ptr()),           \
      N);  

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/4;                                    \
  dim3 block(NT);                                          \
  dim3 grid((S));                                          \
  switch (H)                                               \
  {                                                        \
  case 32:                                                 \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32)         \
    break;                                                 \
  case 64:                                                 \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64)         \
    break;                                                 \
  case 128:                                                \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128)        \
    break;                                                 \
  case 256:                                                \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256)        \
    break;                                                 \
  case 512:                                                \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512)        \
    break;                                                 \
  case 1024:                                               \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024)       \
    break;                                                 \
  case 2048:                                               \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048)       \
    break;                                                 \
  case 4096:                                               \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096)       \
    break;                                                 \
  default:                                                 \
    throw std::runtime_error(                              \
      "only support H: 64/128/.../1024*4");                \
    break;                                                 \
  } 

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)       \
safe_softmax_f16_f32_per_token_kernel<(H)><<<grid, block>>>(  \
      reinterpret_cast<half*>(x.data_ptr()),                  \
      reinterpret_cast<half*>(y.data_ptr()),                  \
      N);  

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H) \
  dim3 block((H));                                           \
  dim3 grid((S));                                            \
  switch ((H))                                               \
  {                                                          \
  case 32:                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32)         \
    break;                                                   \
  case 64:                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64)         \
    break;                                                   \
  case 128:                                                  \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128)        \
    break;                                                   \
  case 256:                                                  \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256)        \
    break;                                                   \
  case 512:                                                  \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512)        \
    break;                                                   \
  case 1024:                                                 \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                                   \
  default:                                                   \
    throw std::runtime_error(                                \
      "only support H: 64/128/256/512/1024");                \
    break;                                                   \
  } 

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)        \
safe_softmax_f16x2_f32_per_token_kernel<(H)/2><<<grid, block>>>( \
      reinterpret_cast<half*>(x.data_ptr()),                     \
      reinterpret_cast<half*>(y.data_ptr()),                     \
      N);  

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/2;                                        \
  dim3 block(NT);                                              \
  dim3 grid((S));                                              \
  switch (H)                                                   \
  {                                                            \
  case 32:                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32)         \
    break;                                                     \
  case 64:                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64)         \
    break;                                                     \
  case 128:                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128)        \
    break;                                                     \
  case 256:                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256)        \
    break;                                                     \
  case 512:                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512)        \
    break;                                                     \
  case 1024:                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                                     \
  case 2048:                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048)       \
    break;                                                     \
  default:                                                     \
    throw std::runtime_error(                                  \
      "only support H: 64/128/.../1024*2");                    \
    break;                                                     \
  } 

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)        \
safe_softmax_f16x8_pack_f32_per_token_kernel<(H)/8><<<grid, block>>>( \
      reinterpret_cast<half*>(x.data_ptr()),                          \
      reinterpret_cast<half*>(y.data_ptr()),                          \
      N);  

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H) \
  const int NT = (H)/8;                                             \
  dim3 block(NT);                                                   \
  dim3 grid((S));                                                   \
  switch (H)                                                        \
  {                                                                 \
  case 32:                                                          \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32)         \
    break;                                                          \
  case 64:                                                          \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64)         \
    break;                                                          \
  case 128:                                                         \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128)        \
    break;                                                          \
  case 256:                                                         \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256)        \
    break;                                                          \
  case 512:                                                         \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512)        \
    break;                                                          \
  case 1024:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024)       \
    break;                                                          \
  case 2048:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048)       \
    break;                                                          \
  case 4096:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096)       \
    break;                                                          \
  case 8192:                                                        \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192)       \
    break;                                                          \
  default:                                                          \
    throw std::runtime_error(                                       \
      "only support H: 64/128/.../1024*8");                         \
    break;                                                          \
  } 

// per token fp32
void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                   
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                            
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                      
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

// per token fp16
void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  // seqlens  
  const int H = x.size(1);  // head size/kv_len
  const int N = S * H; 
  DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)                       
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)                                                                                                                              
  const int S = x.size(0);  
  const int H = x.size(1); 
  const int N = S * H; 
  DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}

// grid memory fence fp32
TORCH_BINDING_SOFTMAX(f32,   torch::kFloat32, float, 1)
TORCH_BINDING_SOFTMAX(f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token)
}
