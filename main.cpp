#include "Config.hpp"
#include "DataLoader.hpp"
#include "PreProcessor.hpp"
#include "TrtInference.hpp"
#include "AnomalyCore.hpp"
#include "PostProcessor.hpp"

// int main() {
//     Config cfg;
//     cudaSetDevice(1); //trt模型里面用的是cuda：1  
//     // 1. 初始化 (加载模型与权重)
//     auto weights = DataLoader::loadBinary(cfg.weight_path, cfg.feat_dim, cfg.pca_dim, cfg.max_patches);
//     TrtInference trt(cfg.engine_path);
//     AnomalyCore core(weights, cfg.feat_dim, cfg.pca_dim, cfg.max_patches);
//     PreProcessor pre(1024, 1024);


//     // 2. 读取图像
//     cv::Mat img = cv::imread("../1.jpg");

//     // 3. 预处理
//     auto pre_res = pre.process(img, cfg.long_edge);

//     float first_pixel = 0;
//     cudaMemcpy(&first_pixel, pre_res.d_input, sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << "Input Tensor [0]: " << first_pixel << " (Should be around -2.0 to 2.0)" << std::endl;

//     // 检查是否输入就有 NaN
//     if (std::isnan(first_pixel)) {
//         std::cerr << "CRITICAL ERROR: Input to TRT is already NaN!" << std::endl;
//     }

//     // 4. 推理 (TRT)
//     int N;
//     float* d_feats = trt.infer(pre_res.d_input, pre_res.new_h, pre_res.new_w, N);

//     // // --- 新增诊断代码 ---
//     // float first_feat = 0;
//     // cudaMemcpy(&first_feat, d_feats, sizeof(float), cudaMemcpyDeviceToHost);
//     // std::cout << "TRT Output [0]: " << first_feat << std::endl;
//     // ------------------

//         // 5. 核心距离计算
//     float* d_dist = core.computeDistance(d_feats, N);

//     // 6. 后处理 (获取结构体结果)
//     AnomalyResult result = PostProcessor::process(d_dist, pre_res.grid_h, pre_res.grid_w, pre_res.original_size, cfg.threshold);


//     //显示得分
//     std::cout<<"score:"<<result.score<<std::endl;

//     // 7. 保存图片
//     cv::imwrite("../mask.png", result.mask);
//     cv::imwrite("../heatmap.jpg", result.heatmap);

//     // 叠加显示示例
//     cv::Mat overlay;
//     cv::addWeighted(img, 0.7, result.heatmap, 0.3, 0, overlay);
//     cv::imwrite("../overlay.jpg", overlay);

//     // 8. 释放内存
//     cudaFree(d_feats);
//     cudaFree(d_dist);
// }

#include <chrono> // 用于计时
#include <cuda_runtime.h>
#include "Config.hpp"
#include "DataLoader.hpp"
#include "PreProcessor.hpp"
#include "TrtInference.hpp"
#include "AnomalyCore.hpp"
#include "PostProcessor.hpp"

// 辅助函数：获取当前时间点
inline auto now() { return std::chrono::high_resolution_clock::now(); }

// 辅助函数：计算毫秒差值
inline double duration(std::chrono::high_resolution_clock::time_point start, 
                      std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    Config cfg;
    
    // 设置设备，所有的同步和计时将针对该设备
    cudaSetDevice(1);  

    // 1. 初始化 (加载模型与权重) - 这一步通常不计入单帧推理时间
    auto weights = DataLoader::loadBinary(cfg.weight_path, cfg.feat_dim, cfg.pca_dim, cfg.max_patches);
    TrtInference trt(cfg.engine_path);
    AnomalyCore core(weights, cfg.feat_dim, cfg.pca_dim, cfg.max_patches);
    PreProcessor pre(1024, 1024);
    for (int i=0;i<10 ; i++) {
        // 读取图像
        cv::Mat img = cv::imread("../1.jpg");
        if (img.empty()) {
            std::cerr << "Failed to read image!" << std::endl;
            return -1;
        }

        std::cout << "\n--- Starting Inference Pipeline ---" << std::endl;

        // --- 全局总耗时起点 ---
        auto total_start = now();

        // 2. 预处理 (OpenCV CPU 运算 + GPU 拷贝)
        auto t1 = now();
        auto pre_res = pre.process(img, cfg.long_edge);
        cudaDeviceSynchronize(); // 等待预处理中的 cudaMemcpy 完成
        auto t2 = now();
        double time_pre = duration(t1, t2);

        // 3. 推理 (TensorRT GPU 推理)
        auto t3 = now();
        int N;
        float* d_feats = trt.infer(pre_res.d_input, pre_res.new_h, pre_res.new_w, N);
        // 注意：TrtInference::infer 内部已经包含了同步，这里可以不重复加，但加上最稳妥
        cudaDeviceSynchronize(); 
        auto t4 = now();
        double time_trt = duration(t3, t4);

        // 4. 核心距离计算 (PCA + L2 距离 GPU Kernels)
        auto t5 = now();
        float* d_dist = core.computeDistance(d_feats, N);
        cudaDeviceSynchronize(); // 必须同步，等待自定义 Kernel 执行完毕
        auto t6 = now();
        double time_core = duration(t5, t6);

        // 5. 后处理 (Resize + Blur + Colormap + GPU->CPU 拷贝)
        auto t7 = now();
        AnomalyResult result = PostProcessor::process(d_dist, pre_res.grid_h, pre_res.grid_w, pre_res.original_size, cfg.threshold);
        cudaDeviceSynchronize(); // 等待数据下载回 CPU
        auto t8 = now();
        double time_post = duration(t7, t8);

        // --- 全局总耗时终点 ---
        auto total_end = now();
        double time_total = duration(total_start, total_end);

        // --- 打印统计结果 ---
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "1. Pre-process Time:   " << time_pre << " ms" << std::endl;
        std::cout << "2. TRT Inference Time: " << time_trt << " ms" << std::endl;
        std::cout << "3. Anomaly Core Time:  " << time_core << " ms" << std::endl;
        std::cout << "4. Post-process Time:  " << time_post << " ms" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Total Pipeline Time:   " << time_total << " ms" << std::endl;
        std::cout << "Anomaly Score:         " << result.score << std::endl;
        
        // 7. 保存图片 (IO 操作不计入推理流水线耗时)
        cv::imwrite("../mask.png", result.mask);
        cv::imwrite("../heatmap.jpg", result.heatmap);

        cv::Mat overlay;
        cv::addWeighted(img, 0.7, result.heatmap, 0.3, 0, overlay);
        cv::imwrite("../overlay.jpg", overlay);

        // 8. 释放内存
        cudaFree(d_feats);
        cudaFree(d_dist);

    }
    

    
    

    return 0;
}