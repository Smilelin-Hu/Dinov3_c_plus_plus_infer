#include "AnomalyCore.hpp"
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

namespace {

constexpr float kNormEps = 1e-10f;
constexpr int kBlockSize1D = 256;
constexpr int kGaussianRadius = 7;
constexpr int kGaussianKernelSize = kGaussianRadius * 2 + 1;

constexpr int kBlurBlockX = 32;
constexpr int kBlurBlockY = 8;
constexpr int kBlurTilePitch = kBlurBlockX + 2 * kGaussianRadius;
constexpr int kBlurTileHeight = kBlurBlockY + 2 * kGaussianRadius;

__constant__ float kGaussianKernel[kGaussianKernelSize] = {
    0.02294906f, 0.03445063f, 0.04858317f, 0.06436224f, 0.08010011f,
    0.09364651f, 0.10285057f, 0.10611540f, 0.10285057f, 0.09364651f,
    0.08010011f, 0.06436224f, 0.04858317f, 0.03445063f, 0.02294906f
};

#ifndef ANOMALY_CORE_DEBUG_VALIDATE
#define ANOMALY_CORE_DEBUG_VALIDATE 0
#endif

inline std::chrono::high_resolution_clock::time_point now() {
    return std::chrono::high_resolution_clock::now();
}

inline double duration(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

inline void checkCublas(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": cuBLAS failure");
    }
}

__global__ void legacyNormalizeAndPCAKernel(const float* feats,
                                            const float* W_row_major,
                                            const float* b,
                                            float* output,
                                            int N,
                                            int F,
                                            int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    float norm_sq = 0.0f;
    for (int k = 0; k < F; ++k) {
        const float val = feats[i * F + k];
        norm_sq += val * val;
    }
    const float inv_norm = rsqrtf(norm_sq + kNormEps);

    for (int j = 0; j < P; ++j) {
        float sum = b[j];
        for (int k = 0; k < F; ++k) {
            sum += (feats[i * F + k] * inv_norm) * W_row_major[k * P + j];
        }
        output[i * P + j] = sum;
    }
}

__global__ void legacyMinL2Kernel(const float* queries,
                                  const float* bank,
                                  float* output,
                                  int N,
                                  int dim,
                                  int bank_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    float min_d = 1e10f;
    const float* q_ptr = queries + i * dim;

    for (int j = 0; j < bank_size; ++j) {
        float dist_sq = 0.0f;
        const float* b_ptr = bank + j * dim;
        for (int k = 0; k < dim; ++k) {
            const float diff = q_ptr[k] - b_ptr[k];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_d) {
            min_d = dist_sq;
        }
    }
    output[i] = sqrtf(min_d);
}

__global__ void rowNormalizeToColumnMajorKernel(const float* feats_row_major,
                                                float* feats_col_major,
                                                int num_rows,
                                                int feat_dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) {
        return;
    }

    __shared__ float shared_sum[kBlockSize1D];
    float local_sum = 0.0f;

    for (int col = tid; col < feat_dim; col += blockDim.x) {
        const float value = feats_row_major[row * feat_dim + col];
        local_sum += value * value;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    const float inv_norm = rsqrtf(shared_sum[0] + kNormEps);
    for (int col = tid; col < feat_dim; col += blockDim.x) {
        feats_col_major[col * num_rows + row] = feats_row_major[row * feat_dim + col] * inv_norm;
    }
}

__global__ void addBiasColumnMajorKernel(float* matrix_col_major,
                                         const float* bias,
                                         int num_rows,
                                         int num_cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_rows * num_cols;
    if (index >= total) {
        return;
    }

    const int col = index / num_rows;
    matrix_col_major[index] += bias[col];
}

__global__ void columnSquaredNormKernel(const float* matrix_col_major,
                                        float* out_sq_norms,
                                        int num_rows,
                                        int num_cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) {
        return;
    }

    __shared__ float shared_sum[kBlockSize1D];
    float local_sum = 0.0f;

    for (int col = tid; col < num_cols; col += blockDim.x) {
        const float value = matrix_col_major[col * num_rows + row];
        local_sum += value * value;
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_sq_norms[row] = shared_sum[0];
    }
}

__global__ void reduceMinDistanceKernel(const float* query_sq_norms,
                                        const float* bank_sq_norms,
                                        const float* dot_products_col_major,
                                        float* min_dist,
                                        int num_queries,
                                        int num_bank) {
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (query_idx >= num_queries) {
        return;
    }

    __shared__ float shared_min[kBlockSize1D];
    float local_min = INFINITY;

    for (int bank_idx = tid; bank_idx < num_bank; bank_idx += blockDim.x) {
        const float dot = dot_products_col_major[query_idx * num_bank + bank_idx];
        float dist_sq = query_sq_norms[query_idx] + bank_sq_norms[bank_idx] - 2.0f * dot;
        dist_sq = fmaxf(dist_sq, 0.0f);
        local_min = fminf(local_min, sqrtf(dist_sq));
    }

    shared_min[tid] = local_min;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_dist[query_idx] = shared_min[0];
    }
}

__global__ void resizeBilinearKernel(const float* src, float* dst,
                                     int src_w, int src_h, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_w && y < dst_h) {
        float src_x = static_cast<float>(x + 0.5f) * (static_cast<float>(src_w) / dst_w) - 0.5f;
        float src_y = static_cast<float>(y + 0.5f) * (static_cast<float>(src_h) / dst_h) - 0.5f;
        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);
        x0 = max(x0, 0);
        y0 = max(y0, 0);
        float dx = src_x - x0;
        float dy = src_y - y0;
        float v00 = src[y0 * src_w + x0];
        float v01 = src[y0 * src_w + x1];
        float v10 = src[y1 * src_w + x0];
        float v11 = src[y1 * src_w + x1];
        dst[y * dst_w + x] = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                             (1.0f - dx) * dy * v10 + dx * dy * v11;
    }
}

__global__ void gaussianBlurVerticalKernel(const float* src, float* dst, int w, int h) {
    __shared__ float tile[kBlurTileHeight][kBlurBlockX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int tile_x = tx;

    for (int tile_y = ty; tile_y < kBlurTileHeight; tile_y += blockDim.y) {
        const int src_y = min(max(blockIdx.y * blockDim.y + tile_y - kGaussianRadius, 0), h - 1);
        if (x < w) {
            tile[tile_y][tile_x] = src[src_y * w + x];
        }
    }
    __syncthreads();

    if (x >= w || y >= h) {
        return;
    }

    const int shared_y = ty + kGaussianRadius;
    float sum = 0.0f;
    for (int k = -kGaussianRadius; k <= kGaussianRadius; ++k) {
        sum += tile[shared_y + k][tile_x] * kGaussianKernel[k + kGaussianRadius];
    }
    dst[y * w + x] = sum;
}

__global__ void gaussianBlurHorizontalKernel(const float* src, float* dst, int w, int h) {
    __shared__ float tile[kBlurBlockY][kBlurTilePitch];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int tile_y = ty;

    for (int tile_x = tx; tile_x < kBlurTilePitch; tile_x += blockDim.x) {
        const int src_x = min(max(blockIdx.x * blockDim.x + tile_x - kGaussianRadius, 0), w - 1);
        if (y < h) {
            tile[tile_y][tile_x] = src[y * w + src_x];
        }
    }
    __syncthreads();

    if (x >= w || y >= h) {
        return;
    }

    const int shared_x = tx + kGaussianRadius;
    float sum = 0.0f;
    for (int k = -kGaussianRadius; k <= kGaussianRadius; ++k) {
        sum += tile[tile_y][shared_x + k] * kGaussianKernel[k + kGaussianRadius];
    }
    dst[y * w + x] = sum;
}

#if ANOMALY_CORE_DEBUG_VALIDATE
void validatePrefix(const float* d_input_features,
                    const float* d_query_pca_col,
                    const float* d_min_dist,
                    const std::vector<float>& weights_W,
                    const std::vector<float>& weights_b,
                    const std::vector<float>& memory_bank,
                    int num_queries,
                    int feat_dim,
                    int pca_dim,
                    int num_bank,
                    cudaStream_t stream) {
    const int rows_to_check = std::min(num_queries, 4);
    if (rows_to_check <= 0) {
        return;
    }

    std::vector<float> host_feats((size_t)rows_to_check * feat_dim);
    std::vector<float> host_pca_row((size_t)rows_to_check * pca_dim);
    std::vector<float> host_pca_col((size_t)rows_to_check * pca_dim);
    std::vector<float> host_min_dist(rows_to_check);

    checkCuda(cudaMemcpyAsync(host_feats.data(),
                              d_input_features,
                              host_feats.size() * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy features for validation");
    checkCuda(cudaMemcpyAsync(host_pca_col.data(),
                              d_query_pca_col,
                              host_pca_col.size() * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy PCA output for validation");
    checkCuda(cudaMemcpyAsync(host_min_dist.data(),
                              d_min_dist,
                              host_min_dist.size() * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy min distances for validation");
    checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize validation stream");

    for (int col = 0; col < pca_dim; ++col) {
        for (int row = 0; row < rows_to_check; ++row) {
            host_pca_row[row * pca_dim + col] = host_pca_col[col * rows_to_check + row];
        }
    }

    const int sample_cols = std::min(pca_dim, 3);
    for (int row = 0; row < rows_to_check; ++row) {
        std::vector<float> normalized(feat_dim);
        float norm_sq = 0.0f;
        for (int k = 0; k < feat_dim; ++k) {
            const float value = host_feats[row * feat_dim + k];
            norm_sq += value * value;
            normalized[k] = value;
        }
        const float inv_norm = 1.0f / std::sqrt(norm_sq + kNormEps);
        for (float& value : normalized) {
            value *= inv_norm;
        }

        std::vector<float> ref_pca(pca_dim, 0.0f);
        float max_pca_diff = 0.0f;
        int max_pca_col = -1;
        for (int col = 0; col < pca_dim; ++col) {
            float accum = weights_b[col];
            for (int k = 0; k < feat_dim; ++k) {
                accum += normalized[k] * weights_W[k * pca_dim + col];
            }
            ref_pca[col] = accum;

            const float gpu_value = host_pca_row[row * pca_dim + col];
            const float diff = std::fabs(gpu_value - accum);
            if (diff > max_pca_diff) {
                max_pca_diff = diff;
                max_pca_col = col;
            }
        }

        float ref_min = INFINITY;
        for (int bank_idx = 0; bank_idx < num_bank; ++bank_idx) {
            float dist_sq = 0.0f;
            for (int col = 0; col < pca_dim; ++col) {
                const float diff = ref_pca[col] - memory_bank[bank_idx * pca_dim + col];
                dist_sq += diff * diff;
            }
            ref_min = std::min(ref_min, std::sqrt(std::max(dist_sq, 0.0f)));
        }

        std::cerr << "[AnomalyCore][validate] row=" << row
                  << " max_pca_diff=" << max_pca_diff
                  << " max_pca_col=" << max_pca_col;
        for (int col = 0; col < sample_cols; ++col) {
            std::cerr << " gpu_pca" << col << "=" << host_pca_row[row * pca_dim + col]
                      << " ref_pca" << col << "=" << ref_pca[col];
        }
        std::cerr << " gpu_min=" << host_min_dist[row]
                  << " ref_min=" << ref_min
                  << std::endl;
    }
}

void validateFastDistancePrefix(const float* d_query_pca_col,
                                const float* d_query_sq_norms,
                                const float* d_dot_products_col,
                                const float* d_min_dist,
                                const std::vector<float>& memory_bank,
                                int num_queries,
                                int pca_dim,
                                int num_bank,
                                cudaStream_t stream) {
    const int rows_to_check = std::min(num_queries, 4);
    const int cols_to_check = std::min(num_bank, 4);
    if (rows_to_check <= 0 || cols_to_check <= 0) {
        return;
    }

    std::vector<float> host_pca((size_t)rows_to_check * pca_dim);
    std::vector<float> host_query_sq(rows_to_check);
    std::vector<float> host_min_dist(rows_to_check);
    std::vector<float> host_dot((size_t)rows_to_check * cols_to_check);
    std::vector<float> host_dot_col((size_t)rows_to_check * cols_to_check);
    std::vector<float> host_pca_tmp((size_t)rows_to_check * pca_dim);

    for (int col = 0; col < pca_dim; ++col) {
        checkCuda(cudaMemcpyAsync(host_pca.data() + (size_t)col * rows_to_check,
                                  d_query_pca_col + (size_t)col * num_queries,
                                  (size_t)rows_to_check * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  "Failed to copy PCA output for fast-distance validation");
    }
    checkCuda(cudaMemcpyAsync(host_query_sq.data(),
                              d_query_sq_norms,
                              host_query_sq.size() * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy query squared norms for fast-distance validation");
    checkCuda(cudaMemcpyAsync(host_min_dist.data(),
                              d_min_dist,
                              host_min_dist.size() * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy min distances for fast-distance validation");
    for (int col = 0; col < cols_to_check; ++col) {
        checkCuda(cudaMemcpyAsync(host_dot_col.data() + (size_t)col * rows_to_check,
                                  d_dot_products_col + (size_t)col * num_queries,
                                  (size_t)rows_to_check * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  "Failed to copy dot products for fast-distance validation");
    }
    checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize fast-distance validation stream");

    for (int row = 0; row < rows_to_check; ++row) {
        for (int col = 0; col < pca_dim; ++col) {
            host_pca_tmp[row * pca_dim + col] = host_pca[(size_t)col * rows_to_check + row];
        }
        for (int bank_idx = 0; bank_idx < cols_to_check; ++bank_idx) {
            host_dot[row * cols_to_check + bank_idx] = host_dot_col[(size_t)bank_idx * rows_to_check + row];
        }
    }
    host_pca.swap(host_pca_tmp);

    const int sample_banks = std::min(cols_to_check, 3);
    for (int row = 0; row < rows_to_check; ++row) {
        std::vector<float> query(pca_dim);
        float ref_query_sq = 0.0f;
        for (int col = 0; col < pca_dim; ++col) {
            const float value = host_pca[row * pca_dim + col];
            query[col] = value;
            ref_query_sq += value * value;
        }

        float ref_min = INFINITY;
        float max_dot_diff = 0.0f;
        int max_dot_bank = -1;
        for (int bank_idx = 0; bank_idx < cols_to_check; ++bank_idx) {
            float ref_dot = 0.0f;
            float ref_dist_sq = 0.0f;
            for (int col = 0; col < pca_dim; ++col) {
                const float bank_value = memory_bank[bank_idx * pca_dim + col];
                ref_dot += query[col] * bank_value;
                const float diff = query[col] - bank_value;
                ref_dist_sq += diff * diff;
            }
            const float gpu_dot = host_dot[row * cols_to_check + bank_idx];
            const float dot_diff = std::fabs(gpu_dot - ref_dot);
            if (dot_diff > max_dot_diff) {
                max_dot_diff = dot_diff;
                max_dot_bank = bank_idx;
            }
            ref_min = std::min(ref_min, std::sqrt(std::max(ref_dist_sq, 0.0f)));
        }

        std::cerr << "[AnomalyCore][fast-distance] row=" << row
                  << " gpu_query_sq=" << host_query_sq[row]
                  << " ref_query_sq=" << ref_query_sq
                  << " max_dot_diff=" << max_dot_diff
                  << " max_dot_bank=" << max_dot_bank;
        for (int bank_idx = 0; bank_idx < sample_banks; ++bank_idx) {
            float ref_dot = 0.0f;
            for (int col = 0; col < pca_dim; ++col) {
                ref_dot += query[col] * memory_bank[bank_idx * pca_dim + col];
            }
            std::cerr << " gpu_dot" << bank_idx << "=" << host_dot[row * cols_to_check + bank_idx]
                      << " ref_dot" << bank_idx << "=" << ref_dot;
        }
        std::cerr << " gpu_min=" << host_min_dist[row]
                  << " ref_min=" << ref_min
                  << std::endl;
    }
}
#endif

}  // namespace

AnomalyCore::AnomalyCore(const ModelWeights& weights, int f_dim, int p_dim, int m_patches)
    : _f_dim(f_dim),
      _p_dim(p_dim),
      _m_patches(m_patches),
      _workspace_capacity(0),
      _stream(nullptr),
      _cublas(nullptr),
      _d_W_col(nullptr),
      _d_b(nullptr),
      _d_bank_col(nullptr),
      _d_bank_sq_norms(nullptr),
      _d_query_norm_col(nullptr),
      _d_query_pca_col(nullptr),
      _d_query_sq_norms(nullptr),
      _d_dot_products_col(nullptr),
      _d_min_dist(nullptr) {
    checkCuda(cudaStreamCreate(&_stream), "Failed to create anomaly-core stream");
    checkCublas(cublasCreate(&_cublas), "Failed to create cuBLAS handle");
    checkCublas(cublasSetStream(_cublas, _stream), "Failed to bind cuBLAS handle to stream");

    checkCuda(cudaMalloc((void**)&_d_W_col, (size_t)_f_dim * _p_dim * sizeof(float)), "Failed to allocate PCA weights");
    checkCuda(cudaMalloc((void**)&_d_b, (size_t)_p_dim * sizeof(float)), "Failed to allocate PCA bias");
    checkCuda(cudaMalloc((void**)&_d_bank_col, (size_t)_m_patches * _p_dim * sizeof(float)), "Failed to allocate memory bank");
    checkCuda(cudaMalloc((void**)&_d_bank_sq_norms, (size_t)_m_patches * sizeof(float)), "Failed to allocate bank norms");

    std::vector<float> weights_col((size_t)_f_dim * _p_dim);
    for (int row = 0; row < _f_dim; ++row) {
        for (int col = 0; col < _p_dim; ++col) {
            weights_col[col * _f_dim + row] = weights.pca_W[row * _p_dim + col];
        }
    }

    std::vector<float> bank_col((size_t)_m_patches * _p_dim);
    std::vector<float> bank_sq_norms((size_t)_m_patches);
    for (int row = 0; row < _m_patches; ++row) {
        float norm_sq = 0.0f;
        for (int col = 0; col < _p_dim; ++col) {
            const float value = weights.memory_bank[row * _p_dim + col];
            bank_col[col * _m_patches + row] = value;
            norm_sq += value * value;
        }
        bank_sq_norms[row] = norm_sq;
    }

    checkCuda(cudaMemcpyAsync(_d_W_col,
                              weights_col.data(),
                              weights_col.size() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream),
              "Failed to upload PCA weights");
    checkCuda(cudaMemcpyAsync(_d_b,
                              weights.pca_b.data(),
                              (size_t)_p_dim * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream),
              "Failed to upload PCA bias");
    checkCuda(cudaMemcpyAsync(_d_bank_col,
                              bank_col.data(),
                              bank_col.size() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream),
              "Failed to upload memory bank");
    checkCuda(cudaMemcpyAsync(_d_bank_sq_norms,
                              bank_sq_norms.data(),
                              bank_sq_norms.size() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream),
              "Failed to upload bank squared norms");
    checkCuda(cudaStreamSynchronize(_stream), "Failed to finish anomaly-core initialization");

#if ANOMALY_CORE_DEBUG_VALIDATE
    _host_pca_W = weights.pca_W;
    _host_pca_b = weights.pca_b;
    _host_memory_bank = weights.memory_bank;
#endif
}

AnomalyCore::~AnomalyCore() {
    if (_d_min_dist) cudaFree(_d_min_dist);
    if (_d_dot_products_col) cudaFree(_d_dot_products_col);
    if (_d_query_sq_norms) cudaFree(_d_query_sq_norms);
    if (_d_query_pca_col) cudaFree(_d_query_pca_col);
    if (_d_query_norm_col) cudaFree(_d_query_norm_col);
    if (_d_bank_sq_norms) cudaFree(_d_bank_sq_norms);
    if (_d_bank_col) cudaFree(_d_bank_col);
    if (_d_b) cudaFree(_d_b);
    if (_d_W_col) cudaFree(_d_W_col);
    if (_cublas) cublasDestroy(_cublas);
    if (_stream) cudaStreamDestroy(_stream);
}

void AnomalyCore::ensureWorkspace(int N) {
    if (N <= _workspace_capacity) {
        return;
    }

    if (_d_min_dist) checkCuda(cudaFree(_d_min_dist), "Failed to free previous min-dist buffer");
    if (_d_dot_products_col) checkCuda(cudaFree(_d_dot_products_col), "Failed to free previous dot buffer");
    if (_d_query_sq_norms) checkCuda(cudaFree(_d_query_sq_norms), "Failed to free previous query-norm buffer");
    if (_d_query_pca_col) checkCuda(cudaFree(_d_query_pca_col), "Failed to free previous PCA col buffer");
    if (_d_query_norm_col) checkCuda(cudaFree(_d_query_norm_col), "Failed to free previous normalized buffer");

    checkCuda(cudaMalloc((void**)&_d_query_norm_col, (size_t)N * _f_dim * sizeof(float)), "Failed to allocate normalized-query workspace");
    checkCuda(cudaMalloc((void**)&_d_query_pca_col, (size_t)N * _p_dim * sizeof(float)), "Failed to allocate PCA col workspace");
    checkCuda(cudaMalloc((void**)&_d_query_sq_norms, (size_t)N * sizeof(float)), "Failed to allocate query squared norms");
    checkCuda(cudaMalloc((void**)&_d_dot_products_col, (size_t)N * _m_patches * sizeof(float)), "Failed to allocate dot-product workspace");
    checkCuda(cudaMalloc((void**)&_d_min_dist, (size_t)N * sizeof(float)), "Failed to allocate min-distance workspace");

    _workspace_capacity = N;
}

float* AnomalyCore::computeDistance(float* d_features, int N) {

    // auto pre_wait_start = now();
    checkCuda(cudaStreamSynchronize(_stream), "Failed to synchronize anomaly stream before computeDistance timing");
    // auto pre_wait_end = now();

    // auto host_enqueue_start = now();
    ensureWorkspace(N);

    cudaEvent_t events[6]{};
    for (cudaEvent_t& event : events) {
        checkCuda(cudaEventCreate(&event), "Failed to create anomaly-core profiling event");
    }

    auto destroyEvents = [&]() {
        cudaError_t first_error = cudaSuccess;
        for (cudaEvent_t& event : events) {
            if (event) {
                cudaError_t status = cudaEventDestroy(event);
                if (first_error == cudaSuccess && status != cudaSuccess) {
                    first_error = status;
                }
                event = nullptr;
            }
        }
        if (first_error != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to destroy anomaly-core profiling event: ") +
                                     cudaGetErrorString(first_error));
        }
    };

    try {
        checkCuda(cudaEventRecord(events[0], _stream), "Failed to record anomaly normalize start event");

    const dim3 normalize_grid(N);
    rowNormalizeToColumnMajorKernel<<<normalize_grid, kBlockSize1D, 0, _stream>>>(
        d_features,
        _d_query_norm_col,
        N,
        _f_dim);
    checkCuda(cudaGetLastError(), "Failed to launch normalize kernel");
    checkCuda(cudaEventRecord(events[1], _stream), "Failed to record anomaly normalize end event");

    const float alpha = 1.0f;
    const float pca_beta = 0.0f;
    checkCublas(cublasSgemm(_cublas,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            N,
                            _p_dim,
                            _f_dim,
                            &alpha,
                            _d_query_norm_col,
                            N,
                            _d_W_col,
                            _f_dim,
                            &pca_beta,
                            _d_query_pca_col,
                            N),
                "Failed to run PCA GEMM");
    checkCuda(cudaEventRecord(events[2], _stream), "Failed to record anomaly PCA GEMM end event");

    const int total_pca_values = N * _p_dim;
    addBiasColumnMajorKernel<<<(total_pca_values + kBlockSize1D - 1) / kBlockSize1D, kBlockSize1D, 0, _stream>>>(
        _d_query_pca_col,
        _d_b,
        N,
        _p_dim);
    checkCuda(cudaGetLastError(), "Failed to launch PCA bias add kernel");
    checkCuda(cudaEventRecord(events[3], _stream), "Failed to record anomaly PCA bias end event");

#if ANOMALY_CORE_DEBUG_VALIDATE
    const float distance_beta = 0.0f;
    checkCublas(cublasSgemm(_cublas,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            _m_patches,
                            N,
                            _p_dim,
                            &alpha,
                            _d_bank_col,
                            _m_patches,
                            _d_query_pca_col,
                            N,
                            &distance_beta,
                            _d_dot_products_col,
                            _m_patches),
                "Failed to run distance GEMM");
    checkCuda(cudaEventRecord(events[4], _stream), "Failed to record anomaly distance GEMM end event");

    columnSquaredNormKernel<<<N, kBlockSize1D, 0, _stream>>>(_d_query_pca_col,
                                                             _d_query_sq_norms,
                                                             N,
                                                             _p_dim);
    checkCuda(cudaGetLastError(), "Failed to launch query-norm kernel");

    reduceMinDistanceKernel<<<N, kBlockSize1D, 0, _stream>>>(_d_query_sq_norms,
                                                             _d_bank_sq_norms,
                                                             _d_dot_products_col,
                                                             _d_min_dist,
                                                             N,
                                                             _m_patches);
    checkCuda(cudaGetLastError(), "Failed to launch min-distance reduction kernel");
    checkCuda(cudaEventRecord(events[5], _stream), "Failed to record anomaly reduction end event");

    validateFastDistancePrefix(_d_query_pca_col,
                               _d_query_sq_norms,
                               _d_dot_products_col,
                               _d_min_dist,
                               _host_memory_bank,
                               N,
                               _p_dim,
                               _m_patches,
                               _stream);
#else
    const float distance_beta = 0.0f;
    checkCublas(cublasSgemm(_cublas,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            _m_patches,
                            N,
                            _p_dim,
                            &alpha,
                            _d_bank_col,
                            _m_patches,
                            _d_query_pca_col,
                            N,
                            &distance_beta,
                            _d_dot_products_col,
                            _m_patches),
                "Failed to run distance GEMM");
    checkCuda(cudaEventRecord(events[4], _stream), "Failed to record anomaly distance GEMM end event");

    columnSquaredNormKernel<<<N, kBlockSize1D, 0, _stream>>>(_d_query_pca_col,
                                                             _d_query_sq_norms,
                                                             N,
                                                             _p_dim);
    checkCuda(cudaGetLastError(), "Failed to launch query-norm kernel");

    reduceMinDistanceKernel<<<N, kBlockSize1D, 0, _stream>>>(_d_query_sq_norms,
                                                             _d_bank_sq_norms,
                                                             _d_dot_products_col,
                                                             _d_min_dist,
                                                             N,
                                                             _m_patches);
    checkCuda(cudaGetLastError(), "Failed to launch min-distance reduction kernel");
    checkCuda(cudaEventRecord(events[5], _stream), "Failed to record anomaly reduction end event");
#endif

#if ANOMALY_CORE_DEBUG_VALIDATE
    validatePrefix(d_features,
                   _d_query_pca_col,
                   _d_min_dist,
                   _host_pca_W,
                   _host_pca_b,
                   _host_memory_bank,
                   N,
                   _f_dim,
                   _p_dim,
                   _m_patches,
                   _stream);
#endif

    checkCuda(cudaStreamSynchronize(_stream), "Failed to finish anomaly-core stream work for profiling");

    destroyEvents();
    } catch (...) {
        destroyEvents();
        throw;
    }

    return _d_min_dist;
}

extern "C" {
void cudaPostProcess(float* d_dist,
                     int gh,
                     int gw,
                     float* d_out_map,
                     float* d_resized,
                     float* d_blur_tmp,
                     int h,
                     int w,
                     float* min_val,
                     float* max_val,
                     cudaStream_t stream
                        ) {

    dim3 block(kBlurBlockX, kBlurBlockY);
    dim3 grid((w + kBlurBlockX - 1) / kBlurBlockX, (h + kBlurBlockY - 1) / kBlurBlockY);

    cudaEvent_t events[5]{};
    for (cudaEvent_t& event : events) {
        checkCuda(cudaEventCreate(&event), "Failed to create postprocess profiling event");
    }

    auto destroyEvents = [&]() {
        cudaError_t first_error = cudaSuccess;
        for (cudaEvent_t& event : events) {
            if (event) {
                cudaError_t status = cudaEventDestroy(event);
                if (first_error == cudaSuccess && status != cudaSuccess) {
                    first_error = status;
                }
                event = nullptr;
            }
        }
        if (first_error != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to destroy postprocess profiling event: ") +
                                     cudaGetErrorString(first_error));
        }
    };

    try {
        checkCuda(cudaEventRecord(events[0], stream), "Failed to record resize start event");
        resizeBilinearKernel<<<grid, block, 0, stream>>>(d_dist, d_resized, gw, gh, w, h);
        checkCuda(cudaGetLastError(), "Failed to launch resize kernel");
        checkCuda(cudaEventRecord(events[1], stream), "Failed to record resize end event");

        gaussianBlurVerticalKernel<<<grid, block, 0, stream>>>(d_resized, d_blur_tmp, w, h);
        checkCuda(cudaGetLastError(), "Failed to launch vertical gaussian kernel");
        checkCuda(cudaEventRecord(events[2], stream), "Failed to record vertical blur end event");

        gaussianBlurHorizontalKernel<<<grid, block, 0, stream>>>(d_blur_tmp, d_out_map, w, h);
        checkCuda(cudaGetLastError(), "Failed to launch horizontal gaussian kernel");
        checkCuda(cudaEventRecord(events[3], stream), "Failed to record horizontal blur end event");

        thrust::device_ptr<float> ptr(d_out_map);
        auto policy = thrust::cuda::par.on(stream);
        auto result = thrust::minmax_element(policy, ptr, ptr + (h * w));
        checkCuda(cudaEventRecord(events[4], stream), "Failed to record minmax end event");
        checkCuda(cudaStreamSynchronize(stream), "Failed to finish postprocess stream work");

        *min_val = *result.first;
        *max_val = *result.second;
        destroyEvents();
    } catch (...) {
        destroyEvents();
        throw;
    }
}
}
