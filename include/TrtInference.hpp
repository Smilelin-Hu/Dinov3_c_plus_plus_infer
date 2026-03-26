#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

class TrtInference {
public:
    TrtInference(const std::string& engine_path);
    void printModelInfo();
    ~TrtInference();

    // 返回 GPU 上的 Feature 指针
    float* infer(float* d_input, int h, int w, int& out_N);
    float* output() const { return d_output; }

private:
    void ensureOutputCapacity(size_t output_elements);

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream;
    float* d_output = nullptr;
    size_t output_capacity = 0;
};