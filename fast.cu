#include <cuda_runtime.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <iostream>
#include <emmintrin.h>
#include <cassert>
#include <climits>
#include <string.h>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <regex>

#define QUERIES_PER_TRIAL (50 * 1000 * 1000)
#define NUM_OF_BLOCKS 1024

const unsigned K=3;

struct LeafEntry {
    int32_t key;
    uint64_t value;
};

void* malloc_huge_cuda(size_t size) {
    void* p;
    cudaMalloc(&p, size);
    return p;
}


inline unsigned pow16(unsigned exponent) {
    // 16^exponent
    return 1<<(exponent<<2);
}

inline unsigned median(unsigned i,unsigned j) {
    return i+(j-1-i)/2;
}


__global__ void searchKernel(const int32_t* v, const int32_t* keys,  int *result, unsigned scale, unsigned index) {
    unsigned key_i = blockIdx.x+NUM_OF_BLOCKS*index;
    if (key_i>=QUERIES_PER_TRIAL)
        return;
    int key_q = keys[key_i];
    const unsigned commonAncesterArray[] = {16,16,16,16,16,16,16,3,1,4,0,5,2,6,16};
    unsigned simd_lane = threadIdx.x;
    unsigned const ancestor = commonAncesterArray[simd_lane];
    __shared__ int child_index;
    __shared__ int shared_gt[16];
    unsigned levelOffset = 0;
    for (int i=0;i<4;++i) {
        size_t addr = ((1 << (4 * i)) - 1) + levelOffset * 16;

//        printf("Thread %d is running\n", idx);
        int32_t v_node = *(v+addr+simd_lane);
//        printf("Thread %d is running after v\n", idx);
        int32_t gt = (key_q>v_node);
        shared_gt[simd_lane] = gt;
//        printf("Thread %d is running after v\n", idx);
        __syncthreads();

        int32_t next_gt = shared_gt[simd_lane+1];
        if (threadIdx.x == 7) {
            if(!gt)
                child_index = 0;
        }
        if (threadIdx.x >= 7 && threadIdx.x<14) {
            if(gt && next_gt==0) {
                child_index = shared_gt[commonAncesterArray[threadIdx.x]]+simd_lane*2-13;
            }
        }
        __syncthreads();
        levelOffset = levelOffset * 16 + child_index;
    }


    unsigned offset = 69904 + levelOffset*scale;
    unsigned pos = levelOffset;
    levelOffset = 0;
    unsigned pageOffset = 0;

    for (int j=0;j<3;++j) {
        size_t addr = offset + (2^(4*j)-1) + levelOffset * 16;
        int32_t v_node = v[addr+simd_lane];

        int32_t gt = (key_q>v_node);
        shared_gt[simd_lane] = gt;
        __syncthreads();

        int32_t next_gt = shared_gt[simd_lane+1];
        if (threadIdx.x == 7 && !gt) {
            child_index = 0;
        }
        if (threadIdx.x >= 7) {
            if(gt && !next_gt) {
                child_index = shared_gt[commonAncesterArray[threadIdx.x]]+simd_lane*2-13;
                if (j==2) {
                    levelOffset = levelOffset * 16 + child_index;
                    int res = ((pos << (4*3)) | levelOffset);
                    result[key_i] = res;
                }
            }
        }
        __syncthreads();
        levelOffset = levelOffset * 16 + child_index;
    }

}

void cudaSearch(std::vector<int>& queries, const int32_t* fast, unsigned scale, int* res) {
    int* deviceData;
    int sizeInByte = queries.size()*sizeof(int);
    cudaMalloc(&deviceData, sizeInByte);
    cudaMemcpy(deviceData, queries.data(), sizeInByte, cudaMemcpyHostToDevice);
    const int numStreams = 648;
    int * check;
    cudaMalloc(&check, sizeInByte);

    cudaStream_t streams[numStreams];
    for(int i=0;i<numStreams;++i) {
        cudaStreamCreate(&streams[i]);
    }

    for(size_t i=0;i<queries.size()/NUM_OF_BLOCKS;++i) {
        int streamIndex = i % numStreams;
        int key = queries[i];
        searchKernel<<<NUM_OF_BLOCKS, 15, 0, streams[streamIndex]>>>(fast, deviceData, check, scale, i);
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
//        cudaStreamDestroy(streams[i]);
    }

//    uint64_t hostCheck = 0;
    cudaMemcpy(res, check, sizeInByte, cudaMemcpyDeviceToHost);

//    for (int i=0;i<queries.size();++i) {
//        hostCheck += check[i];
//    }

    for (int i = 0; i < numStreams; ++i) {
//        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(check);
    cudaFree(deviceData);
}



int32_t * pinCuda(int32_t *fast, unsigned n) {
    void* add = (malloc_huge_cuda(n*sizeof(int32_t)));
    cudaMemcpy(add, fast, sizeof(int) * n, cudaMemcpyHostToDevice);
    return (int32_t *)add;
}


