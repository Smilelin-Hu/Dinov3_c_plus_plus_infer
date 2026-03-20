#include "PreProcessor.hpp"
#include <cuda_runtime.h>

PreProcessor::PreProcessor(int max_h, int max_w) {
    // 预分配足够的 GPU 空间 (1 * 3 * max_h * max_w)
    cudaMalloc(&_d_input, 1 * 3 * max_h * max_w * sizeof(float));
}

PreProcessor::~PreProcessor() {
    cudaFree(_d_input);
}

// PreProcessResult PreProcessor::process(const cv::Mat& frame, int long_edge) {
//     PreProcessResult res;
//     res.original_size = frame.size();

//     // 1. 颜色空间转换 BGR -> RGB
//     cv::Mat rgb;
//     cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

//     // 2. 计算缩放尺寸 (保证 patch_size=16 对齐)
//     float scale = (float)long_edge / std::max(rgb.rows, rgb.cols);
//     res.new_h = (int(rgb.rows * scale) / 16) * 16;
//     res.new_w = (int(rgb.cols * scale) / 16) * 16;

//     res.new_h = std::max(res.new_h, 16);
//     res.new_w = std::max(res.new_w, 16);

//     res.grid_h = res.new_h / 16;
//     res.grid_w = res.new_w / 16;

//     std::cout<<"这是宽"<<res.new_w<<std::endl;
//     std::cout<<"这是高"<<res.new_h<<std::endl;

//     // 3. Resize
//     cv::Mat resized;
//     cv::resize(rgb, resized, cv::Size(res.new_w, res.new_h), 0, 0, cv::INTER_LINEAR);

//     // 4. 归一化 (Pixel / 255.0 - Mean) / Std
//     resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    
//     // ImageNet 均值和方差
//     cv::Scalar mean(0.485, 0.456, 0.406);
//     cv::Scalar std(0.229, 0.224, 0.225);
//     cv::subtract(resized, mean, resized);
//     cv::divide(resized, std, resized);

//     // 5. HWC (OpenCV) 转换为 CHW (TensorRT)
//     // 使用 OpenCV split 将三个通道分开
//     std::vector<cv::Mat> channels(3);
//     cv::split(resized, channels);

//     size_t plane_size = (size_t)res.new_h * res.new_w * sizeof(float);
    
//     // 将三个通道的数据依次拷贝到 GPU 内存的不同位置
//     // _d_input[0...plane] = R, _d_input[plane...2*plane] = G, _d_input[2*plane...3*plane] = B
//     cudaMemcpy(_d_input, channels[0].ptr<float>(), plane_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(_d_input + (res.new_h * res.new_w), channels[1].ptr<float>(), plane_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(_d_input + (2 * res.new_h * res.new_w), channels[2].ptr<float>(), plane_size, cudaMemcpyHostToDevice);

//     res.d_input = _d_input;
//     return res;
// }

PreProcessResult PreProcessor::process(const cv::Mat& frame, int long_edge) {
    PreProcessResult res;
    res.original_size = frame.size();

    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    // 计算尺寸
    float scale = (float)long_edge / std::max(rgb.rows, rgb.cols);
    res.new_h = (int(rgb.rows * scale) / 16) * 16;
    res.new_w = (int(rgb.cols * scale) / 16) * 16;
    
    res.grid_h = res.new_h / 16;
    res.grid_w = res.new_w / 16;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(res.new_w, res.new_h), 0, 0, cv::INTER_LINEAR);

    // --- 强化归一化逻辑 ---
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    
    // 分通道处理，防止 Scalar 广播错误
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[] = {0.229f, 0.224f, 0.225f};
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    // 拷贝到 GPU (NCHW 布局)
    size_t plane_size = (size_t)res.new_h * res.new_w * sizeof(float);
    for (int i = 0; i < 3; ++i) {
        // 确保内存连续
        cv::Mat cont_chan = channels[i].isContinuous() ? channels[i] : channels[i].clone();
        cudaMemcpy(_d_input + i * res.new_h * res.new_w, cont_chan.ptr<float>(), plane_size, cudaMemcpyHostToDevice);
    }

    res.d_input = _d_input;
    return res;
}