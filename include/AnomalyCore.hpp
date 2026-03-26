#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include "DataLoader.hpp"

#ifdef __cplusplus
extern "C" {
#endif

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
                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif

class AnomalyCore {
public:
    AnomalyCore(const ModelWeights& weights, int f_dim, int p_dim, int m_patches);
    ~AnomalyCore();

    float* computeDistance(float* d_features, int N);
    cudaStream_t stream() const { return _stream; }

private:
    void ensureWorkspace(int N);

    int _f_dim;
    int _p_dim;
    int _m_patches;
    int _workspace_capacity;

    cudaStream_t _stream;
    cublasHandle_t _cublas;

    float* _d_W_col;
    float* _d_b;
    float* _d_bank_col;
    float* _d_bank_sq_norms;

    float* _d_query_norm_col;
    float* _d_query_pca_col;
    float* _d_query_sq_norms;
    float* _d_dot_products_col;
    float* _d_min_dist;

#if ANOMALY_CORE_DEBUG_VALIDATE
    std::vector<float> _host_pca_W;
    std::vector<float> _host_pca_b;
    std::vector<float> _host_memory_bank;
#endif
};
