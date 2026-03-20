#include "PostProcessor.hpp"
#include "AnomalyCore.hpp" // 必须包含这个头文件，它里面有 cudaPostProcess 的正确声明
#include <cuda_runtime.h>
#include <iostream>

AnomalyResult PostProcessor::process(float* d_dist, int gh, int gw, cv::Size ori_size, float threshold) {
    int h = ori_size.height;
    int w = ori_size.width;

    // 1. 在 GPU 上分配最终热力图的空间 (使用显式类型转换)
    float* d_final_map = nullptr;
    cudaMalloc((void**)&d_final_map, (size_t)h * w * sizeof(float));

    // 2. 调用原生 CUDA 实现的后处理
    // 注意：这里传递的是 &min_v 和 &max_v (取地址)，以匹配 float* 指针参数
    float min_v = 0.0f;
    float max_v = 0.0f;
    cudaPostProcess(d_dist, gh, gw, d_final_map, h, w, &min_v, &max_v);

    // 3. 将处理好的热力图数据从 GPU 下载到 CPU
    cv::Mat h_map(h, w, CV_32FC1);
    cudaMemcpy(h_map.data, d_final_map, (size_t)h * w * sizeof(float), cudaMemcpyDeviceToHost);

    // 4. 在 CPU 上进行最后的归一化 (这部分计算量极小)
    cv::Mat norm_map = (h_map - min_v) / (max_v - min_v + 1e-8);

    // 5. 组装结果结构体 AnomalyResult
    AnomalyResult res;
    res.score = max_v; // 异常得分

    // --- 生成二值 Mask ---
    cv::Mat mask_float;
    cv::threshold(norm_map, mask_float, threshold, 255.0, cv::THRESH_BINARY);
    mask_float.convertTo(res.mask, CV_8UC1);

    // --- 生成彩色热力图 Heatmap ---
    cv::Mat heatmap_8u;
    norm_map.convertTo(heatmap_8u, CV_8UC1, 255.0); // 线性映射到 0-255
    cv::applyColorMap(heatmap_8u, res.heatmap, cv::COLORMAP_JET); // 应用 JET 伪彩色图

    // 6. 释放 GPU 临时显存
    if (d_final_map) {
        cudaFree(d_final_map);
    }

    return res;
}