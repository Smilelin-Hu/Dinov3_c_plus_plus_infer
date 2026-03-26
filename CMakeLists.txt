cmake_minimum_required(VERSION 3.10)
project(anomaly LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# 在文件开头附近添加（project 命令之后 忽略警告）
cmake_policy(SET CMP0074 NEW)

# 设置CUDA路径
set(CUDA_ROOT "/usr/local/cuda-12.8")
set(CUDART_LIB "${CUDA_ROOT}/lib64/libcudart.so")
set(CUBLAS_LIB "${CUDA_ROOT}/lib64/libcublas.so")

# 设置TensorRT路径
set(TensorRT_ROOT "/home/ps/train/HJL/TensorRT-10.15.1.29")
set(NVINFER_LIB "${TensorRT_ROOT}/lib/libnvinfer.so")
set(NVONNXPARSER_LIB "${TensorRT_ROOT}/lib/libnvonnxparser.so")

# 设置ONNX Runtime路径
set(ONNXRUNTIME_ROOT "/home/ps/train/HJL/onnxruntime-linux-x64-gpu-1.17.3")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

# 检查ONNX Runtime库是否存在
if(NOT EXISTS ${ONNXRUNTIME_LIB})
    message(FATAL_ERROR "ONNX Runtime library not found: ${ONNXRUNTIME_LIB}")
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

# 查找 CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 定义onnxruntime主程序的源文件
set(MAIN_SOURCES
    main.cpp
    src/DataLoader.cpp
    src/PostProcessor.cpp
    src/TrtInference.cpp
    src/AnomalyCore.cu
    src/PreProcessor.cpp    
)

# 开启 CUDA 支持
enable_language(CUDA)

# 设置CUDA编译器标志（重要）
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14 -O3 -DANOMALY_CORE_DEBUG_VALIDATE=1)

# 包含所有头文件路径
include_directories(
    ${TensorRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${CUDA_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(main ${MAIN_SOURCES}) # 改用 cuda_add_executable

###########################################################
# 主程序 - predict 版本
###########################################################

target_link_libraries(main
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDART_LIB}
    ${CUBLAS_LIB}
    ${CUDA_LIBRARIES}
    cudart
)

# 设置 RPATH
set_target_properties(main PROPERTIES
    BUILD_RPATH "${OpenCV_DIR}/lib:${ONNXRUNTIME_ROOT}/lib:${TensorRT_ROOT}/lib:${CUDA_ROOT}/lib64"
    INSTALL_RPATH "${OpenCV_DIR}/lib:${ONNXRUNTIME_ROOT}/lib:${TensorRT_ROOT}/lib:${CUDA_ROOT}/lib64"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86 
)
