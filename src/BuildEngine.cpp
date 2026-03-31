#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>

using namespace nvinfer1;

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if(severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

int main(int argc,char** argv)
{
    Logger logger;
  
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <onnx_file> <engine_file>" << std::endl;
        return -1;
    }
    
    const std::string onnx_file = argv[1];
    const std::string engine_file = argv[2];

    // builder
    TrtUniquePtr<IBuilder> builder(createInferBuilder(logger));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder." << std::endl;
        return -1;
    }

    uint32_t flag = 1U << (uint32_t)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;

    TrtUniquePtr<INetworkDefinition> network(builder->createNetworkV2(flag));
    if (!network) {
        std::cerr << "Failed to create TensorRT network." << std::endl;
        return -1;
    }

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser." << std::endl;
        return -1;
    }

    // parse onnx
    if(!parser->parseFromFile(onnx_file.c_str(),
        (int)ILogger::Severity::kWARNING))
    {
        std::cout<<"ONNX parse failed"<<std::endl;
        return -1;
    }

    // builder config
    TrtUniquePtr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create TensorRT builder config." << std::endl;
        return -1;
    }

    config->setMemoryPoolLimit(
        MemoryPoolType::kWORKSPACE,
        2ULL << 30
    );

    // config->setFlag(BuilderFlag::kFP16); //精度异常太低了，默认是32位的

    // build engine
    TrtUniquePtr<ICudaEngine> engine(
        builder->buildEngineWithConfig(*network, *config));

    // 判断引擎是否创建成功    
    if (engine == nullptr) {
        std::cerr << "TensorRT engine build failed (engine is null)." << std::endl;
        return -1;
    }
    
    // serialize
    TrtUniquePtr<IHostMemory> serialized(engine->serialize());

    //判断序列化是否成功
    if (serialized == nullptr) {
        std::cerr << "TensorRT engine serialize failed (serialized is null)." << std::endl;
        return -1;
    }

    std::ofstream file(engine_file, std::ios::binary);

    file.write(
        (char*)serialized->data(),
        serialized->size()
    );

    file.close();

    std::cout<<"TensorRT engine saved: "<<engine_file<<std::endl;

    return 0;
}