#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>

struct AnomalyResult {
    cv::Mat mask;
    cv::Mat heatmap;
    std::vector<std::vector<cv::Point>> polygons;
    float score;
};

class PostProcessor {
public:
    PostProcessor();
    ~PostProcessor();

    AnomalyResult process(float* d_dist,
                          int gh,
                          int gw,
                          cv::Size ori_size,
                          float threshold,
                          cudaStream_t stream);

private:
    void ensureCapacity(cv::Size ori_size);
    void release();

    cv::Size _capacity;
    float* _d_final_map;
    float* _d_resized;
    float* _d_blur_tmp;
    float* _host_map;
    cv::Mat _norm_map;
    cv::Mat _heatmap_8u;
};
