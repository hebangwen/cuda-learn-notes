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
#define MAX_EXP_F32  88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)


// -------------------------------------- FP32 -------------------------------------- 
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/256), block(K=256) 
__global__ void sigmoid_f32_kernel(float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
      // 注意这后面添加的后缀 f，表明这些函数针对的是 float 类型，如果没有 f，则是 double 类型
      float val = fmaxf(fminf(x[tid], MAX_EXP_F32), MIN_EXP_F32);
      float sig = 1 / (1 + expf(-val));
      y[tid] = sig;
    }
    // [END MANUAL IMPLEMENTATION]
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/256), block(256/4)
__global__ void sigmoid_f32x4_kernel(float* x, float* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = tid << 2;
    if (gid < N) {
      float reg_x[4], reg_sig[4];
      LDST128BITS(reg_x[0]) = LDST128BITS(x[gid]);
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float clip = fmaxf(fminf(reg_x[i], MAX_EXP_F32), MIN_EXP_F32);
        reg_sig[i] = 1 / (1 + expf(-clip));
      }

      LDST128BITS(y[gid]) = LDST128BITS(reg_sig[0]);
    }
    // [END MANUAL IMPLEMENTATION]
}

// -------------------------------------- FP16 -------------------------------------- 
__global__ void sigmoid_f16_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    // 使用 hexp 和 __hmax __hmin , operator< > 也可以用
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const half one = __float2half(1.0);
    if (tid < N) {
      half val = __hmax(__hmin(x[tid], MAX_EXP_F16), MIN_EXP_F16);
      // 注意：这条语句编译会产生报错，不支持 int + __half 的操作符
      // half sig = 1 / (1 + hexp(-val));
      // y[tid] = sig;
      half sig = one / (one + hexp(-val));
      y[tid] = sig;
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void sigmoid_f16x2_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = tid << 1;
    const half one = __float2half(1.0);
    if (gid < N) {
      half2 reg_x = HALF2(x[gid]);
      reg_x.x = __hmax(__hmin(reg_x.x, MAX_EXP_F16), MIN_EXP_F16);
      reg_x.y = __hmax(__hmin(reg_x.y, MAX_EXP_F16), MIN_EXP_F16);

      half2 reg_sig;
      reg_sig.x = one / (one + hexp(-reg_x.x));
      reg_sig.y = one / (one + hexp(-reg_x.y));

      HALF2(y[gid]) = reg_sig;
    }
    // [END MANUAL IMPLEMENTATION]
}

// unpack f16x8
__global__ void sigmoid_f16x8_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = tid << 3;
    const half one = __float2half(1.0f);
    if (gid < N) {
      half2 reg_x[4], reg_y[4];
  #pragma unroll
      for (int i = 0; i < 4; i++) {
          reg_x[i] = HALF2(x[gid + (i << 1)]);

          reg_x[i].x = __hmax(__hmin(reg_x[i].x, MAX_EXP_F16), MIN_EXP_F16);
          reg_x[i].y = __hmax(__hmin(reg_x[i].y, MAX_EXP_F16), MIN_EXP_F16);
          reg_y[i].x = one / (one + hexp(-reg_x[i].x));
          reg_y[i].y = one / (one + hexp(-reg_x[i].y));
      }

      HALF2(y[gid + 0]) = reg_y[0];
      HALF2(y[gid + 2]) = reg_y[1];
      HALF2(y[gid + 4]) = reg_y[2];
      HALF2(y[gid + 6]) = reg_y[3];
    }
    // [END MANUAL IMPLEMENTATION]
}

// pack f16x8
__global__ void sigmoid_f16x8_pack_kernel(half* x, half* y, int N) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = tid << 3;
    const half one = __float2half(1.0f);
    if (gid < N) {
      half2 reg_x[4], reg_y[4];
      LDST128BITS(reg_x[0]) = LDST128BITS(x[gid]);
  #pragma unroll
      for (int i = 0; i < 4; i++) {
          reg_x[i].x = __hmax(__hmin(reg_x[i].x, MAX_EXP_F16), MIN_EXP_F16);
          reg_x[i].y = __hmax(__hmin(reg_x[i].y, MAX_EXP_F16), MIN_EXP_F16);
          reg_y[i].x = one / (one + hexp(-reg_x[i].x));
          reg_y[i].y = one / (one + hexp(-reg_x[i].y));
      }

      LDST128BITS(y[gid]) = LDST128BITS(reg_y[0]);
    }
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

#define TORCH_BINDING_SIGMOID(packed_type, th_type, element_type, n_elements)    \
void sigmoid_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                         \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                         \
  const int ndim = x.dim();                                                      \
  if (ndim != 2) {                                                               \
    int N = 1;                                                                   \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                           \
    dim3 block(256 / (n_elements));                                              \
    dim3 grid((N + 256 - 1) / 256);                                              \
    sigmoid_##packed_type##_kernel<<<grid, block>>>(                             \
      reinterpret_cast<element_type*>(x.data_ptr()),                             \
      reinterpret_cast<element_type*>(y.data_ptr()), N);                         \
  } else {                                                                       \
    const int S = x.size(0);                                                     \
    const int K = x.size(1);                                                     \
    const int N = S * K;                                                         \
    if ((K/(n_elements)) <= 1024) {                                              \
      dim3 block(K/(n_elements));                                                \
      dim3 grid(S);                                                              \
      sigmoid_##packed_type##_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type*>(x.data_ptr()),                           \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                       \
    } else {                                                                     \
      int N = 1;                                                                 \
      for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                         \
      dim3 block(256 / (n_elements));                                            \
      dim3 grid((N + 256 - 1) / 256);                                            \
      sigmoid_##packed_type##_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type*>(x.data_ptr()),                           \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                       \
    }                                                                            \
  }                                                                              \
}


TORCH_BINDING_SIGMOID(f32,        torch::kFloat32,    float,    1)
TORCH_BINDING_SIGMOID(f32x4,      torch::kFloat32,    float,    4)
TORCH_BINDING_SIGMOID(f16,        torch::kHalf,       half,     1)
TORCH_BINDING_SIGMOID(f16x2,      torch::kHalf,       half,     2)
TORCH_BINDING_SIGMOID(f16x8,      torch::kHalf,       half,     8)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}
