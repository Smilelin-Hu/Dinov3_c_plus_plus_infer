#include "PostProcessor.hpp"
#include "AnomalyCore.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {
inline void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}
}

PostProcessor::PostProcessor()
    : _capacity(0, 0),
      _d_final_map(nullptr),
      _d_resized(nullptr),
      _d_blur_tmp(nullptr),
      _host_map(nullptr) {}

PostProcessor::~PostProcessor() {
    release();
}

void PostProcessor::ensureCapacity(cv::Size ori_size) {
    if (ori_size.width <= _capacity.width && ori_size.height <= _capacity.height) {
        return;
    }

    release();

    _capacity = ori_size;
    const size_t elems = static_cast<size_t>(_capacity.width) * _capacity.height;
    checkCuda(cudaMalloc((void**)&_d_final_map, elems * sizeof(float)),
              "Failed to allocate final anomaly map");
    checkCuda(cudaMalloc((void**)&_d_resized, elems * sizeof(float)),
              "Failed to allocate resized anomaly map");
    checkCuda(cudaMalloc((void**)&_d_blur_tmp, elems * sizeof(float)),
              "Failed to allocate blur workspace");
    checkCuda(cudaMallocHost((void**)&_host_map, elems * sizeof(float)),
              "Failed to allocate pinned anomaly map host buffer");
    _norm_map.create(_capacity, CV_32FC1);
    _heatmap_8u.create(_capacity, CV_8UC1);
}

void PostProcessor::release() {
    _norm_map.release();
    _heatmap_8u.release();
    if (_host_map) {
        cudaFreeHost(_host_map);
        _host_map = nullptr;
    }
    if (_d_blur_tmp) {
        cudaFree(_d_blur_tmp);
        _d_blur_tmp = nullptr;
    }
    if (_d_resized) {
        cudaFree(_d_resized);
        _d_resized = nullptr;
    }
    if (_d_final_map) {
        cudaFree(_d_final_map);
        _d_final_map = nullptr;
    }
    _capacity = cv::Size(0, 0);
}

AnomalyResult PostProcessor::process(float* d_dist,
                                     int gh,
                                     int gw,
                                     cv::Size ori_size,
                                     float threshold,
                                     cudaStream_t stream
                                     ) {
    const int h = ori_size.height;
    const int w = ori_size.width;
    ensureCapacity(ori_size);

    // if (profile) {
    //     profile->pre_wait_ms = 0.0;
    //     profile->gpu_host_ms = 0.0;
    //     profile->d2h_sync_ms = 0.0;
    //     profile->cpu_norm_mask_subtract_ms = 0.0;
    //     profile->cpu_norm_mask_normalize_ms = 0.0;
    //     profile->cpu_norm_mask_threshold_ms = 0.0;
    //     profile->cpu_norm_mask_convert_ms = 0.0;
    //     profile->cpu_norm_mask_ms = 0.0;
    //     profile->cpu_heatmap_convert_ms = 0.0;
    //     profile->cpu_heatmap_colormap_ms = 0.0;
    //     profile->cpu_heatmap_ms = 0.0;
    //     profile->cpu_ms = 0.0;
    // }

    // auto pre_wait_start = now();
    checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize stream before postprocess timing");
    // auto pre_wait_end = now();

    // auto gpu_start = now();
    float min_v = 0.0f;
    float max_v = 0.0f;
    cudaPostProcess(d_dist, gh, gw, _d_final_map, _d_resized, _d_blur_tmp, h, w, &min_v, &max_v, stream);
    // auto gpu_end = now();

    // auto d2h_start = now();
    checkCuda(cudaMemcpyAsync(_host_map,
                              _d_final_map,
                              static_cast<size_t>(h) * w * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              stream),
              "Failed to copy anomaly map to host");
    checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize postprocess stream");
    // auto d2h_end = now();

    // auto cpu_start = now();
    const float scale = 1.0f / (max_v - min_v + 1e-8f);

    cv::Mat h_map(h, w, CV_32FC1, _host_map);
    cv::Mat norm_map = _norm_map(cv::Rect(0, 0, w, h));
    cv::Mat heatmap_8u = _heatmap_8u(cv::Rect(0, 0, w, h));

    AnomalyResult res;
    res.score = max_v;

    // auto cpu_norm_mask_start = now();
    // auto cpu_norm_mask_subtract_start = now();
    h_map.convertTo(norm_map, CV_32FC1, scale, -min_v * scale);
    // auto cpu_norm_mask_subtract_end = now();
    // auto cpu_norm_mask_normalize_start = now();
    // auto cpu_norm_mask_normalize_end = now();
    // auto cpu_norm_mask_threshold_start = now();
    const float raw_threshold = threshold * (max_v - min_v + 1e-8f) + min_v;
    cv::compare(h_map, cv::Scalar(raw_threshold), res.mask, cv::CMP_GT);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(res.mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    res.polygons = std::move(contours);
    // auto cpu_norm_mask_threshold_end = now();
    // auto cpu_norm_mask_convert_start = now();
    // auto cpu_norm_mask_convert_end = now();
    // auto cpu_norm_mask_end = now();

    // auto cpu_heatmap_start = now();
    // auto cpu_heatmap_convert_start = now();
    norm_map.convertTo(heatmap_8u, CV_8UC1, 255.0);
    // auto cpu_heatmap_convert_end = now();
    // auto cpu_heatmap_colormap_start = now();
    cv::applyColorMap(heatmap_8u, res.heatmap, cv::COLORMAP_JET);
    // auto cpu_heatmap_colormap_end = now();
    // auto cpu_heatmap_end = now();
    // auto cpu_end = now();

    // const double pre_wait_ms = duration(pre_wait_start, pre_wait_end);
    // const double gpu_host_ms = duration(gpu_start, gpu_end);
    // const double d2h_sync_ms = duration(d2h_start, d2h_end);
    // const double cpu_norm_mask_subtract_ms = duration(cpu_norm_mask_subtract_start, cpu_norm_mask_subtract_end);
    // const double cpu_norm_mask_normalize_ms = duration(cpu_norm_mask_normalize_start, cpu_norm_mask_normalize_end);
    // const double cpu_norm_mask_threshold_ms = duration(cpu_norm_mask_threshold_start, cpu_norm_mask_threshold_end);
    // const double cpu_norm_mask_convert_ms = duration(cpu_norm_mask_convert_start, cpu_norm_mask_convert_end);
    // const double cpu_norm_mask_ms = duration(cpu_norm_mask_start, cpu_norm_mask_end);
    // const double cpu_heatmap_convert_ms = duration(cpu_heatmap_convert_start, cpu_heatmap_convert_end);
    // const double cpu_heatmap_colormap_ms = duration(cpu_heatmap_colormap_start, cpu_heatmap_colormap_end);
    // const double cpu_heatmap_ms = duration(cpu_heatmap_start, cpu_heatmap_end);
    // const double cpu_ms = duration(cpu_start, cpu_end);

    // if (profile) {
    //     profile->pre_wait_ms = pre_wait_ms;
    //     profile->gpu_host_ms = gpu_host_ms;
    //     profile->d2h_sync_ms = d2h_sync_ms;
    //     profile->cpu_norm_mask_subtract_ms = cpu_norm_mask_subtract_ms;
    //     profile->cpu_norm_mask_normalize_ms = cpu_norm_mask_normalize_ms;
    //     profile->cpu_norm_mask_threshold_ms = cpu_norm_mask_threshold_ms;
    //     profile->cpu_norm_mask_convert_ms = cpu_norm_mask_convert_ms;
    //     profile->cpu_norm_mask_ms = cpu_norm_mask_ms;
    //     profile->cpu_heatmap_convert_ms = cpu_heatmap_convert_ms;
    //     profile->cpu_heatmap_colormap_ms = cpu_heatmap_colormap_ms;
    //     profile->cpu_heatmap_ms = cpu_heatmap_ms;
    //     profile->cpu_ms = cpu_ms;
    // }

    // std::cout << "    post.pre_wait: " << pre_wait_ms << " ms, "
    //           << "post.gpu_host: " << gpu_host_ms << " ms, "
    //           << "post.resize: " << gpu_profile.resize_ms << " ms, "
    //           << "post.blur_v: " << gpu_profile.blur_vertical_ms << " ms, "
    //           << "post.blur_h: " << gpu_profile.blur_horizontal_ms << " ms, "
    //           << "post.minmax: " << gpu_profile.minmax_ms << " ms, "
    //           << "post.d2h_sync: " << d2h_sync_ms << " ms, "
    //           << "post.cpu_norm_mask.normgen: " << cpu_norm_mask_subtract_ms << " ms, "
    //           << "post.cpu_norm_mask.normalize: " << cpu_norm_mask_normalize_ms << " ms, "
    //           << "post.cpu_norm_mask.threshold: " << cpu_norm_mask_threshold_ms << " ms, "
    //           << "post.cpu_norm_mask.convert: " << cpu_norm_mask_convert_ms << " ms, "
    //           << "post.cpu_norm_mask: " << cpu_norm_mask_ms << " ms, "
    //           << "post.cpu_heatmap.convert: " << cpu_heatmap_convert_ms << " ms, "
    //           << "post.cpu_heatmap.colormap: " << cpu_heatmap_colormap_ms << " ms, "
    //           << "post.cpu_heatmap: " << cpu_heatmap_ms << " ms, "
    //           << "post.cpu: " << cpu_ms << " ms" << std::endl;

    return res;
}
