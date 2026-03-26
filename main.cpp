#include "Config.hpp"
#include "DataLoader.hpp"
#include "PreProcessor.hpp"
#include "TrtInference.hpp"
#include "AnomalyCore.hpp"
#include "PostProcessor.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>

inline auto now() { return std::chrono::high_resolution_clock::now(); }

inline double duration(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    try {
        Config cfg;
        cudaSetDevice(1);

        auto weights = DataLoader::loadBinary(cfg.weight_path, cfg.feat_dim, cfg.pca_dim, cfg.max_patches);
        TrtInference trt(cfg.engine_path);
        AnomalyCore core(weights, cfg.feat_dim, cfg.pca_dim, cfg.max_patches);
        PreProcessor pre(1024, 1024);
        PostProcessor post;

        for (int i = 0; i < 5; ++i) {
            cv::Mat img = cv::imread("../1.jpg");
            if (img.empty()) {
                std::cerr << "Failed to read image!" << std::endl;
                return -1;
            }

            std::cout << "\n--- Starting Inference Pipeline ---" << std::endl;
            auto total_start = now();

            auto t1 = now();
            auto pre_res = pre.process(img, cfg.long_edge);
            auto t2 = now();
            double time_pre = duration(t1, t2);

            auto t3 = now();
            int N = 0;
            float* d_feats = trt.infer(pre_res.d_input, pre_res.new_h, pre_res.new_w, N);
            auto t4 = now();
            double time_trt = duration(t3, t4);

            const int expected_tokens = pre_res.grid_h * pre_res.grid_w;
            if (N != expected_tokens) {
                throw std::runtime_error("TensorRT patch token count does not match preprocessing grid");
            }

            auto t5 = now();
            float* d_dist = core.computeDistance(d_feats, N);
            auto t6 = now();
            double time_core = duration(t5, t6);

            auto t7 = now();
            AnomalyResult result = post.process(
                d_dist,
                pre_res.grid_h,
                pre_res.grid_w,
                pre_res.original_size,
                cfg.threshold,
                core.stream());
            auto t8 = now();
            double time_post = duration(t7, t8);

            auto total_end = now();
            double time_total = duration(total_start, total_end);

            std::cout << std::fixed << std::setprecision(3);
            // std::cout << "    pre.h2d: " << pre_profile.h2d_ms << " ms" << std::endl;
            // std::cout << "    trt.output_alloc: " << trt_profile.output_alloc_ms << " ms, "
            //           << "trt.bind: " << trt_profile.bind_ms << " ms, "
            //           << "trt.enqueue_host: " << trt_profile.enqueue_host_ms << " ms, "
            //           << "trt.sync_host: " << trt_profile.sync_host_ms << " ms" << std::endl;
            // std::cout << "    core.pre_wait: " << core_profile.pre_wait_ms << " ms, "
            //           << "core.enqueue_host: " << core_profile.host_enqueue_ms << " ms, "
            //           << "core.normalize: " << core_profile.normalize_ms << " ms, "
            //           << "core.pca_gemm: " << core_profile.pca_gemm_ms << " ms, "
            //           << "core.pca_bias: " << core_profile.pca_bias_ms << " ms, "
            //           << "core.distance_gemm: " << core_profile.distance_gemm_ms << " ms, "
            //           << "core.reduce: " << core_profile.reduce_ms << " ms" << std::endl;
            std::cout << "1. Pre-process Time:   " << time_pre << " ms" << std::endl;
            std::cout << "2. TRT Inference Time: " << time_trt << " ms" << std::endl;
            std::cout << "3. Anomaly Core Time:  " << time_core << " ms" << std::endl;
            std::cout << "4. Post-process Time:  " << time_post << " ms" << std::endl;
            std::cout << "-----------------------------------" << std::endl;
            std::cout << "Total Pipeline Time:   " << time_total << " ms" << std::endl;
            std::cout << "Anomaly Score:         " << result.score << std::endl;

            cv::imwrite("../mask.png", result.mask);
            cv::imwrite("../heatmap.jpg", result.heatmap);

            cv::Mat overlay;
            cv::addWeighted(img, 0.6, result.heatmap, 0.4, 0, overlay);
            cv::imwrite("../overlay.jpg", overlay);

            cv::Mat polygon_overlay = img.clone();
            cv::polylines(polygon_overlay, result.polygons, true, cv::Scalar(0, 0, 255), 2);
            cv::imwrite("../polygon_overlay.jpg", polygon_overlay);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Pipeline failed: " << ex.what() << std::endl;
        return -1;
    }
}
