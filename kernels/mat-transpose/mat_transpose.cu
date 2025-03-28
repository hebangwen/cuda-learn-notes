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

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// -------------------------------------- FP32 --------------------------------------
// col2row means read x[row][col] and write y[col][row]
// row2col means read x[col][row] and write y[row][col]
__global__ void mat_transpose_f32_col2row_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int N = row * col;
    if (tid < N) {
      int yrow = tid % col;
      int ycol = tid / col;
      y[yrow * row + ycol] = x[tid];
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32_row2col_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int N = row * col;
    if (tid < N) {
      int xrow = tid % row;
      int xcol = tid / row;
      y[tid] = x[xrow * col + xcol];
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_col2row_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = tid << 2;
    int N = row * col;
    if (gid < N) {
      float4 rx = FLOAT4(x[gid]);
      int yrow = gid % col;
      int ycol = gid / col;
      y[yrow * row + ycol] = rx.x;
      y[yrow * row + ycol + row] = rx.y;
      y[yrow * row + ycol + row * 2] = rx.z;
      y[yrow * row + ycol + row * 3] = rx.w;
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_row2col_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = tid << 2;
    int N = row * col;
    if (gid < N) {
      float4 ry;
      int xrow = gid % row;
      int xcol = gid / row;
      ry.x = x[xrow * col + xcol];
      ry.y = x[xrow * col + xcol + col];
      ry.z = x[xrow * col + xcol + col * 2];
      ry.w = x[xrow * col + xcol + col * 3];
      FLOAT4(y[gid]) = ry;
    }
    // [END MANUAL IMPLEMENTATION]
}

// work for row == col
__global__ void mat_transpose_f32_diagonal2d_kernel(
  float *x, float *y, int row, int col) {
    // [START MANUAL IMPLEMENTATION]
    int xcol = threadIdx.x + blockIdx.x * blockDim.x;
    int xrow = threadIdx.y + blockIdx.y * blockDim.y;
    if (xcol < col && xrow < row) {
      y[xcol * row + xrow] = x[xrow * col + xcol];
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32_col2row2d_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int xcol = threadIdx.x + blockIdx.x * blockDim.x;
    int xrow = threadIdx.y + blockIdx.y * blockDim.y;
    if (xcol < col && xrow < row) {
      y[xcol * row + xrow] = x[xrow * col + xcol];
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32_row2col2d_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int yrow = threadIdx.x + blockIdx.x * blockDim.x;
    int ycol = threadIdx.y + blockIdx.y * blockDim.y;
    if (ycol < row && yrow < col) {
      y[yrow * row + ycol] = x[ycol * col + yrow];
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_col2row2d_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int xcol = (threadIdx.x + blockIdx.x * blockDim.x) << 2;
    int xrow = threadIdx.y + blockIdx.y * blockDim.y;
    if (xcol < col && xrow < row) {
      float4 rx = FLOAT4(x[xrow * col + xcol]);
      y[xcol * row + xrow] = rx.x;
      y[(xcol + 1) * row + xrow] = rx.y;
      y[(xcol + 2) * row + xrow] = rx.z;
      y[(xcol + 3) * row + xrow] = rx.w;
    }
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_row2col2d_kernel(
  float *x, float *y, const int row, const int col) {
    // [START MANUAL IMPLEMENTATION]
    int yrow = threadIdx.x + blockIdx.x * blockDim.x;
    int ycol = (threadIdx.y + blockIdx.y * blockDim.y) << 2;
    if (ycol < row && yrow < col) {
      float4 ry;
      ry.x = x[ycol * col + yrow];
      ry.y = x[(ycol + 1) * col + yrow];
      ry.z = x[(ycol + 2) * col + yrow];
      ry.w = x[(ycol + 3) * col + yrow];
      FLOAT4(y[yrow * row + ycol]) = ry;
    }
    // [END MANUAL IMPLEMENTATION]
}


__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(
  float *x, float *y, const int row, const int col){
    // [START MANUAL IMPLEMENTATION]
    // 每个子线程读取 4 个 float 到 shared memory 中
    // 等 block 内读取完毕之后，同步
    // 从共享内存中取出被转置的 4 个 float
    // 写回到 y 中
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_shared_row2col2d_kernel(
  float *x, float *y, const int row, const int col){
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(
  float *x, float *y, const int row, const int col){
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

__global__ void mat_transpose_f32x4_shared_bcf_row2col2d_kernel(
  float *x, float *y, const int row, const int col){
    // [START MANUAL IMPLEMENTATION]
    // TODO: 请在此实现内核代码
    // [END MANUAL IMPLEMENTATION]
}

// TODO: may support double buffer pipeline mat transpose ?
// TODO: may support fp16 mat transpose ?

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type)))                    \
  {                                                            \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack) \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y)            \
  {                                                                     \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                              \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                              \
    const int M = x.size(0);                                            \
    const int N = x.size(1);                                            \
    dim3 block(WARP_SIZE);                                              \
    dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));          \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                      \
        reinterpret_cast<element_type *>(x.data_ptr()),                 \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);          \
  }

#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type, n_element_row, n_element_col) \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y)                                \
  {                                                                                             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                      \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                                      \
    const int M = x.size(0);                                                                    \
    const int N = x.size(1);                                                                    \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                                       \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),                            \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));                           \
    mat_transpose_##tag##2d_kernel <<<grid, block>>>(                                           \
      reinterpret_cast<element_type *>(x.data_ptr()),                                           \
      reinterpret_cast<element_type *>(y.data_ptr()), M, N);                                    \
  }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2d index. easier for diagonal 
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)
// 2d 索引启动时是按照 / n_element_row 的，是否不太对
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)
// diagonal index method.
TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)
// shared memory
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_row2col, torch::kFloat32, float, 4, 1)
// shared memory with bcf
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_row2col, torch::kFloat32, float, 4, 1)
// TODO: may support double buffer pipeline mat transpose ?
// TODO: may support fp16 mat transpose ?


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col2d)
  // diagonal index method.
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_diagonal2d)
  // shared memory optimize
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_row2col2d)
  //shared memory optimize with bcf
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_shared_bcf_row2col2d)
}
