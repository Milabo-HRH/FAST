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


__global__ void searchKernel(const int32_t* v, int32_t key_q, unsigned long long int *result, unsigned scale) {
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
                    unsigned long long int res = ((pos << 4) | levelOffset);
                    atomicAdd(result, res);
                }
            }
        }
        __syncthreads();
        levelOffset = levelOffset * 16 + child_index;
    }

}

unsigned long long int cudaSearch(std::vector<int>& queries, const int32_t* fast, unsigned scale) {
    const int numStreams = 6480;
    unsigned long long int * check;
    cudaMalloc(reinterpret_cast<void **>(&check), sizeof(unsigned long long int));

    cudaStream_t streams[numStreams];
    for(int i=0;i<numStreams;++i) {
        cudaStreamCreate(&streams[i]);
    }

    for(size_t i=0;i<queries.size();++i) {
        int streamIndex = i % numStreams;
        int key = queries[i];
        searchKernel<<<1, 15, 0, streams[streamIndex]>>>(fast, key, check, scale);
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
//        cudaStreamDestroy(streams[i]);
    }

    uint64_t hostCheck = 0;
    cudaMemcpy(&hostCheck, check, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numStreams; ++i) {
//        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(check);
    return hostCheck;
}



int32_t * pinCuda(int32_t *fast, unsigned n) {
    void* add = (malloc_huge_cuda(n*sizeof(int32_t)));
    cudaMemcpy(add, fast, sizeof(int) * n, cudaMemcpyHostToDevice);
    return (int32_t *)add;
}


