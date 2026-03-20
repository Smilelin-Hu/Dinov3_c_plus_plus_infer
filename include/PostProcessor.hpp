#pragma once
#include <opencv2/opencv.hpp>

// 1. 定义结果结构体，包含二值图、彩色热力图和异常得分
struct AnomalyResult {
    cv::Mat mask;      // 二值化后的掩码图 (CV_8UC1)
    cv::Mat heatmap;   // 伪彩色热力图 (CV_8UC3)
    float score;       // 整张图的最高异常得分
};

class PostProcessor {
public:
    /**
     * @brief 执行后处理：缩放、高斯平滑、归一化、生成热力图和 Mask
     * 
     * @param d_dist    GPU 上的原始距离向量指针 (来自 AnomalyCore)
     * @param gh        特征图高度 (Grid Height)
     * @param gw        特征图宽度 (Grid Width)
     * @param ori_size  原始图像尺寸 (用于 Resize)
     * @param threshold 异常判定的阈值 (通常为 0.0 - 1.0 之间)
     * @return          返回包含所有结果的 AnomalyResult 结构体
     */
    static AnomalyResult process(float* d_dist, int gh, int gw, cv::Size ori_size, float threshold);
};