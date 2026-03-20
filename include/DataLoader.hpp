#pragma once
#include <vector>
#include <string>

struct ModelWeights {
    std::vector<float> pca_W;     // [feat_dim * pca_dim]
    std::vector<float> pca_b;     // [pca_dim]
    std::vector<float> memory_bank; // [max_patches * pca_dim]
};

class DataLoader {
public:
    // 从二进制文件读取权重 (Python端需用 .to_numpy().tofile() 保存)
    static ModelWeights loadBinary(const std::string& path, int f_dim, int p_dim, int m_patches);
};