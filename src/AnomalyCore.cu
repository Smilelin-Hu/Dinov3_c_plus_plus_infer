#include "AnomalyCore.hpp"
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <iostream>
#include <math.h>

// =========================================================
// 1. 所有的 CUDA Kernels (核心算法)
// =========================================================

__global__ void normalizeAndPCAKernel(const float* feats, const float* W, const float* b, 
    float* output, int N, int F, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 第 i 个 patch
    if (i >= N) return;

    // 1. 对原始特征进行 L2 归一化 (对应 Python 的 F.normalize)
    float norm_sq = 0;
    for (int k = 0; k < F; ++k) {
    float val = feats[i * F + k];
    norm_sq += val * val;
    }
    float inv_norm = rsqrtf(norm_sq + 1e-10f);

    // 2. PCA 投影: out = (feat * inv_norm) @ W + b
    for (int j = 0; j < P; ++j) {
    float sum = b[j];
    for (int k = 0; k < F; ++k) {
    // 注意：W 的布局是 [F, P]，行优先读取
    sum += (feats[i * F + k] * inv_norm) * W[k * P + j];
    }
    output[i * P + j] = sum;
}
}


// 计算到 Memory Bank 的最小 L2 距离
__global__ void minL2Kernel(const float* queries, const float* bank, float* output, 
    int N, int dim, int bank_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float min_d = 1e10f;
    const float* q_ptr = queries + i * dim;

    for (int j = 0; j < bank_size; ++j) 
    {
        float dist_sq = 0;
        const float* b_ptr = bank + j * dim;
        for (int k = 0; k < dim; ++k) 
        {
            float diff = q_ptr[k] - b_ptr[k];
            dist_sq += diff * diff;
        }
    // 使用平方和可以加快速度，最后再开方
        if (dist_sq < min_d) min_d = dist_sq;
    }
    output[i] = sqrtf(min_d);
}


// 双线性插值缩放 (用于空间还原)
__global__ void resizeBilinearKernel(const float* src, float* dst, 
                                     int src_w, int src_h, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_w && y < dst_h) {
        float src_x = (float)(x + 0.5f) * ((float)src_w / dst_w) - 0.5f;
        float src_y = (float)(y + 0.5f) * ((float)src_h / dst_h) - 0.5f;
        int x0 = (int)floorf(src_x); int y0 = (int)floorf(src_y);
        int x1 = min(x0 + 1, src_w - 1); int y1 = min(y0 + 1, src_h - 1);
        x0 = max(x0, 0); y0 = max(y0, 0);
        float dx = src_x - x0; float dy = src_y - y0;
        float v00 = src[y0 * src_w + x0]; float v01 = src[y0 * src_w + x1];
        float v10 = src[y1 * src_w + x0]; float v11 = src[y1 * src_w + x1];
        dst[y * dst_w + x] = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                             (1.0f - dx) * dy * v10 + dx * dy * v11;
    }
}

// 高斯模糊 (15x15 kernel, sigma=4.0)
__global__ void gaussianBlurKernel(const float* src, float* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        float sum = 0, weight_sum = 0;
        float sigma = 4.0f; int radius = 7; 
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int nx = min(max(x + j, 0), w - 1);
                int ny = min(max(y + i, 0), h - 1);
                float weight = expf(-(float)(i * i + j * j) / (2.0f * sigma * sigma));
                sum += src[ny * w + nx] * weight;
                weight_sum += weight;
            }
        }
        dst[y * w + x] = sum / weight_sum;
    }
}

// =========================================================
// 2. AnomalyCore 类方法实现 (必须带 AnomalyCore:: 前缀)
// =========================================================

// 类方法实现
AnomalyCore::AnomalyCore(const ModelWeights& w, int f_dim, int p_dim, int m_patches) 
    : _f_dim(f_dim), _p_dim(p_dim), _m_patches(m_patches) {
    cudaMalloc((void**)&_d_W, (size_t)f_dim * p_dim * sizeof(float));
    cudaMalloc((void**)&_d_b, (size_t)p_dim * sizeof(float));
    cudaMalloc((void**)&_d_bank, (size_t)m_patches * p_dim * sizeof(float));

    cudaMemcpy(_d_W, w.pca_W.data(), (size_t)f_dim * p_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, w.pca_b.data(), (size_t)p_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_bank, w.memory_bank.data(), (size_t)m_patches * p_dim * sizeof(float), cudaMemcpyHostToDevice);
}

AnomalyCore::~AnomalyCore() {
    if (_d_W) cudaFree(_d_W);
    if (_d_b) cudaFree(_d_b);
    if (_d_bank) cudaFree(_d_bank);
}

// 计算异常距离
// float* AnomalyCore::computeDistance(float* d_features, int N) {
//     int patch_N = N;
//     float* d_patch_ptr = d_features;

//     float* d_pca_out;
//     cudaMalloc((void**)&d_pca_out, (size_t)patch_N * _p_dim * sizeof(float));

//     normalizeAndPCAKernel<<<(patch_N + 255) / 256, 256>>>(
//         d_patch_ptr, _d_W, _d_b, d_pca_out, patch_N, _f_dim, _p_dim
//     );

//     float* d_dist_out;
//     cudaMalloc((void**)&d_dist_out, (size_t)patch_N * sizeof(float));
//     minL2Kernel<<<(patch_N + 255) / 256, 256>>>(
//         d_pca_out, _d_bank, d_dist_out, patch_N, _p_dim, _m_patches
//     );

//     cudaFree(d_pca_out);
//     return d_dist_out;
// }

float* AnomalyCore::computeDistance(float* d_features, int N) {
    float* d_pca_out;
    cudaMalloc((void**)&d_pca_out, (size_t)N * _p_dim * sizeof(float));

    // 1. PCA + Normalize
    normalizeAndPCAKernel<<<(N + 255) / 256, 256>>>(d_features, _d_W, _d_b, d_pca_out, N, _f_dim, _p_dim);


    float* d_dist_out;
    cudaMalloc((void**)&d_dist_out, (size_t)N * sizeof(float));
    minL2Kernel<<<(N + 255) / 256, 256>>>(d_pca_out, _d_bank, d_dist_out, N, _p_dim, _m_patches);

    cudaFree(d_pca_out);
    return d_dist_out;
}


// =========================================================
// 3. 供 PostProcessor 调用的 C 包装函数
// =========================================================

extern "C" {
    void cudaPostProcess(float* d_dist, int gh, int gw, float* d_out_map, int h, int w, float* min_val, float* max_val) {
        float* d_resized;
        cudaMalloc((void**)&d_resized, (size_t)h * w * sizeof(float));
        
        dim3 block(16, 16);
        dim3 grid((w + 15) / 16, (h + 15) / 16);

        // 执行后处理 Kernel
        resizeBilinearKernel<<<grid, block>>>(d_dist, d_resized, gw, gh, w, h);
        gaussianBlurKernel<<<grid, block>>>(d_resized, d_out_map, w, h);

        // 使用 Thrust 求最大最小值
        thrust::device_ptr<float> ptr(d_out_map);
        auto result = thrust::minmax_element(ptr, ptr + (h * w));
        *min_val = *result.first;
        *max_val = *result.second;

        cudaFree(d_resized);
    }
}