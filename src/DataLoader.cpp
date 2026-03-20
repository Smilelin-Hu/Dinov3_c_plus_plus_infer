#include "DataLoader.hpp"
#include <fstream>
#include <iostream>

ModelWeights DataLoader::loadBinary(const std::string& path, int f_dim, int p_dim, int m_patches) {
    ModelWeights w;
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }

    // 计算每个部分的大小
    size_t size_W = (size_t)f_dim * p_dim;
    size_t size_b = (size_t)p_dim;
    size_t size_bank = (size_t)m_patches * p_dim;

    w.pca_W.resize(size_W);
    w.pca_b.resize(size_b);
    w.memory_bank.resize(size_bank);

    // 顺序读取二进制数据
    // 注意：Python 端保存的顺序必须与此处一致
    file.read(reinterpret_cast<char*>(w.pca_W.data()), size_W * sizeof(float));
    file.read(reinterpret_cast<char*>(w.pca_b.data()), size_b * sizeof(float));
    file.read(reinterpret_cast<char*>(w.memory_bank.data()), size_bank * sizeof(float));

    std::cout << "First weight of PCA_W: " << w.pca_W[0] << std::endl;
    std::cout << "First weight of MemoryBank: " << w.memory_bank[0] << std::endl;
    if (file.gcount() == 0 && size_bank > 0) {
        std::cerr << "Warning: Read 0 bytes for memory bank. Check file size." << std::endl;
    }

    file.close();
    std::cout << "Successfully loaded weights from " << path << std::endl;
    return w;
}