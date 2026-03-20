#include "TrtInference.hpp"
#include <fstream>
#include <iostream>
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) printf("TRT: %s\n", msg);
    }
} gLogger;

TrtInference::TrtInference(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(data.data(), data.size());
    context = engine->createExecutionContext();
    cudaStreamCreate(&stream);
    printModelInfo();
}

void TrtInference::printModelInfo() {
    int nbIO = engine->getNbIOTensors(); // 获取输入输出张量的总数
    std::cout << "===== TensorRT Model IO Info =====" << std::endl;
    for (int i = 0; i < nbIO; ++i) {
        const char* name = engine->getIOTensorName(i); // 获取张量名称
        nvinfer1::DataType type = engine->getTensorDataType(name);
        std::cout << "  - DataType: " << (int)type << " (0:Float, 1:Half, 2:Int8, 3:Int32)" << std::endl;
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name); // 判断是输入还是输出
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            std::cout << "Input Tensor Name: " << name << std::endl;
        } else {
            std::cout << "Output Tensor Name: " << name << std::endl;
            // 顺便打印一下输出的维度，看看是不是 [1, N, 384]
            auto dims = engine->getTensorShape(name);
            std::cout << "  - Shape: [";
            for(int j=0; j<dims.nbDims; ++j) std::cout << dims.d[j] << (j==dims.nbDims-1 ? "" : ",");
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << "==================================" << std::endl;
}

float* TrtInference::infer(float* d_input, int h, int w, int& out_N) {
    // 1. 设置输入尺寸 (必须和 Engine 一致)
    context->setInputShape("input", nvinfer1::Dims4{1, 3, h, w});
    
    // 2. 使用正确的输出张量名称 "patch_tokens"
    const char* out_name = "patch_tokens"; 
    auto out_dims = context->getTensorShape(out_name);
    
    out_N = out_dims.d[1];
    int out_C = out_dims.d[2];

    float* d_output;
    cudaMalloc((void**)&d_output, out_N * out_C * sizeof(float));

    // 3. 绑定地址
    context->setTensorAddress("input", d_input);
    context->setTensorAddress(out_name, d_output); // 这里改成 out_name

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    return d_output;
}

TrtInference::~TrtInference() {
    if (context) {
        delete context; // TensorRT 10 使用标准的 delete
        context = nullptr;
    }
    if (engine) {
        delete engine;
        engine = nullptr;
    }
    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
    }
}