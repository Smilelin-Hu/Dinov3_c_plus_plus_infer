#pragma once
#include <string>

struct Config {
    // 模型参数
    int patch_size = 16;
    int feat_dim = 384;   // DINOv3 ViT-S 为 384
    int pca_dim = 384;    // 降维后的维度
    int max_patches = 12000; // Memory Bank 大小

    // 推理参数
    int long_edge = 896;
    float threshold = 0.85f;

    // 路径
    std::string engine_path = "../dinov3_simfp32.engine";
    std::string weight_path = "../weights.bin"; // 需从Python导出为二进制
};