#include "TrtInference.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) printf("TRT: %s\n", msg);
    }
} gLogger;

inline void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

inline void checkTrt(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}
}

TrtInference::TrtInference(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(data.data(), data.size());
    context = engine->createExecutionContext();
    checkCuda(cudaStreamCreate(&stream), "Failed to create TensorRT stream");
    printModelInfo();
}

void TrtInference::printModelInfo() {
    int nbIO = engine->getNbIOTensors();
    std::cout << "===== TensorRT Model IO Info =====" << std::endl;
    for (int i = 0; i < nbIO; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::DataType type = engine->getTensorDataType(name);
        std::cout << "  - DataType: " << (int)type << " (0:Float, 1:Half, 2:Int8, 3:Int32)" << std::endl;
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            std::cout << "Input Tensor Name: " << name << std::endl;
        } else {
            std::cout << "Output Tensor Name: " << name << std::endl;
            auto dims = engine->getTensorShape(name);
            std::cout << "  - Shape: [";
            for (int j = 0; j < dims.nbDims; ++j) std::cout << dims.d[j] << (j == dims.nbDims - 1 ? "" : ",");
            std::cout << "]" << std::endl;
        }
    }

    std::cout << "==================================" << std::endl;
}

void TrtInference::ensureOutputCapacity(size_t output_elements) {
    if (output_elements <= output_capacity) {
        return;
    }

    if (d_output) {
        checkCuda(cudaFree(d_output), "Failed to free TensorRT output buffer");
        d_output = nullptr;
    }

    checkCuda(cudaMalloc((void**)&d_output, output_elements * sizeof(float)),
              "Failed to allocate TensorRT output");
    output_capacity = output_elements;
}

float* TrtInference::infer(float* d_input, int h, int w, int& out_N) {
    // if (profile) {
    //     profile->enqueue_host_ms = 0.0;
    //     profile->sync_host_ms = 0.0;
    //     profile->output_alloc_ms = 0.0;
    //     profile->bind_ms = 0.0;
    // }

    checkTrt(context->setInputShape("input", nvinfer1::Dims4{1, 3, h, w}), "Failed to set TensorRT input shape");

    const char* out_name = "patch_tokens";
    auto out_dims = context->getTensorShape(out_name);
    if (out_dims.nbDims != 3 || out_dims.d[1] <= 0 || out_dims.d[2] <= 0) {
        throw std::runtime_error("Invalid TensorRT output shape for patch_tokens");
    }

    out_N = out_dims.d[1];
    const int out_C = out_dims.d[2];
    const size_t output_elements = static_cast<size_t>(out_N) * out_C;

    // auto output_alloc_start = now();
    ensureOutputCapacity(output_elements);
    // auto output_alloc_end = now();

    // auto bind_start = now();
    checkTrt(context->setTensorAddress("input", d_input), "Failed to bind TensorRT input tensor");
    checkTrt(context->setTensorAddress(out_name, d_output), "Failed to bind TensorRT output tensor");
    // auto bind_end = now();

    // auto enqueue_start = now();
    checkTrt(context->enqueueV3(stream), "Failed to enqueue TensorRT inference");
    // auto enqueue_end = now();

    // auto sync_start = now();
    checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize TensorRT stream");
    // auto sync_end = now();

    // if (profile) {
    //     profile->output_alloc_ms = duration(output_alloc_start, output_alloc_end);
    //     profile->bind_ms = duration(bind_start, bind_end);
    //     profile->enqueue_host_ms = duration(enqueue_start, enqueue_end);
    //     profile->sync_host_ms = duration(sync_start, sync_end);
    // }

    return output();
}

TrtInference::~TrtInference() {
    if (context) {
        delete context;
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
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}
