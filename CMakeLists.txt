cmake_minimum_required(VERSION 3.16.3)
project(FAST LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_GENERATOR_PLATFORM x64)

set(SOURCE_FILES main.cpp FAST.cu)


add_executable(FAST ${SOURCE_FILES})

target_compile_options(FAST PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Ofast -march=native -flto -pedantic -Wall -Wextra -Weffc++>
)

# 为 CUDA 设置编译选项，注意，这里我们不使用 -march=native 和 -flto
target_compile_options(FAST PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# 可选：为 CUDA 代码开启分离编译
set_target_properties(FAST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

