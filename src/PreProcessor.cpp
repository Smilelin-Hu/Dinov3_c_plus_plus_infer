#include "PreProcessor.hpp"
#include <algorithm>
#include <cuda_runtime.h>

PreProcessor::PreProcessor(int max_h, int max_w) {
    cudaMalloc(&_d_input, 1 * 3 * max_h * max_w * sizeof(float));
}

PreProcessor::~PreProcessor() {
    cudaFree(_d_input);
}

PreProcessResult PreProcessor::process(const cv::Mat& frame, int long_edge) {
 
    PreProcessResult res;
    res.original_size = frame.size();

    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    constexpr int patch_size = 16;
    const float scale = static_cast<float>(long_edge) / std::max(rgb.rows, rgb.cols);
    res.new_h = std::max((static_cast<int>(rgb.rows * scale) / patch_size) * patch_size, patch_size);
    res.new_w = std::max((static_cast<int>(rgb.cols * scale) / patch_size) * patch_size, patch_size);

    res.grid_h = res.new_h / patch_size;
    res.grid_w = res.new_w / patch_size;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(res.new_w, res.new_h), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};

    const size_t plane_size = static_cast<size_t>(res.new_h) * res.new_w * sizeof(float);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
        cv::Mat cont_chan = channels[i].isContinuous() ? channels[i] : channels[i].clone();
        cudaMemcpy(_d_input + static_cast<size_t>(i) * res.new_h * res.new_w,
                   cont_chan.ptr<float>(),
                   plane_size,
                   cudaMemcpyHostToDevice);
    }

    res.d_input = _d_input;
    return res;
}
