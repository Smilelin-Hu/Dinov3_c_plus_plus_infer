#pragma once
#include <cuda_runtime.h>
#include "DataLoader.hpp"

// 使用 __cplusplus 宏确保 C/C++ 兼容性
// 这里包裹的是供 PostProcessor.cpp 调用的 CUDA 包装函数
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CUDA 后处理包装函数
 * @param d_dist    GPU 上的原始距离向量 [N]
 * @param gh        特征图高度 (grid_h, 如 56)
 * @param gw        特征图宽度 (grid_w, 如 30)
 * @param d_out_map GPU 上的输出热力图 Buffer [h * w]
 * @param h         原始图像高度 (如 896)
 * @param w         原始图像宽度 (如 480)
 * @param min_val   用于接收 Map 中的最小值 (指针形式以兼容 extern "C")
 * @param max_val   用于接收 Map 中的最大值 (指针形式以兼容 extern "C")
 */
void cudaPostProcess(float* d_dist, int gh, int gw, float* d_out_map, int h, int w, float* min_val, float* max_val);

#ifdef __cplusplus
}
#endif

class AnomalyCore {
public:
    /**
     * @brief 构造函数：分配并拷贝 GPU 权重
     */
    AnomalyCore(const ModelWeights& weights, int f_dim, int p_dim, int m_patches);
    
    /**
     * @brief 析构函数：释放 GPU 权重内存
     */
    ~AnomalyCore();

    /**
     * @brief 核心距离计算函数
     * @param d_features 来自 TensorRT 的原始特征指针 [N, f_dim]
     * @param N          特征点的总数 (如 56 * 30 = 1680)
     * @return           返回 GPU 上的距离数组指针 [N]
     */
    float* computeDistance(float* d_features, int N);

private:
    int _f_dim;      // 原始特征维度 (如 384)
    int _p_dim;      // PCA 降维后的维度 (如 384)
    int _m_patches;  // Memory Bank 的 Patch 数量 (如 12000)
    
    // GPU 权重指针
    float *_d_W;     // PCA 权重矩阵 [f_dim, p_dim]
    float *_d_b;     // PCA 偏置向量 [p_dim]
    float *_d_bank;  // Memory Bank 矩阵 [m_patches, p_dim]

    // 注意：这里不再需要 cublasHandle_t
};