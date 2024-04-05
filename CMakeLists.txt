cmake_minimum_required(VERSION 3.16.3)
project(FAST LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_GENERATOR_PLATFORM x64)

# Find Boost package, specifically looking for the Asio component
#find_package(Boost REQUIRED COMPONENTS system)

set(SOURCE_FILES main.cpp fast.cu)

add_executable(FAST ${SOURCE_FILES})

target_compile_options(FAST PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Ofast -march=native -flto -pedantic -Wall -Wextra -Weffc++>
)

target_compile_options(FAST PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# Link Boost libraries to the FAST target
#target_link_libraries(FAST PRIVATE Boost::system)

# Optional: Enable separate compilation for CUDA code
set_target_properties(FAST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
