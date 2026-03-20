#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

struct PreProcessResult {
    float* d_input;        // GPU 上的输入 Buffer
    int new_h, new_w;
    int grid_h, grid_w;
    cv::Size original_size;
};

class PreProcessor {
public:
    PreProcessor(int max_h, int max_w);
    ~PreProcessor();
    PreProcessResult process(const cv::Mat& frame, int long_edge);

private:
    float* _d_input; // 预分配 GPU 空间
};