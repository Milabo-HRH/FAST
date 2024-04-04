/*
  Fast Architecture Sensitive Tree layout for binary search trees
  (Kim et. al, SIGMOD 2010)

  implementation by Ruihuan He, UofT, 2024

  notes:
  -keys are 4 byte integers
  -SSE instructions are used for comparisons
  -huge memory pages (2MB)
  -page blocks store 4 levels of cacheline blocks
  -cacheline blocks store 15 keys and are 64-byte aligned
  -the parameter K results in a tree size of (2^(16+K*4))
 */
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
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
#include <atomic>
#include <thread>
#include <chrono>

const unsigned K=3;

struct LeafEntry {
    int32_t key;
    uint64_t value;
};

void* malloc_huge(size_t size) {
    void* p=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
#if __linux__
    madvise(p,size,MADV_HUGEPAGE);
#endif
    return p;
}

inline unsigned pow16(unsigned exponent) {
    // 16^exponent
    return 1<<(exponent<<2);
}

inline unsigned pow64(unsigned exponent) {
    // 64^exponent
    // Since 64 is 2^6, we can left shift 6 times per exponent
    return 1 << (exponent * 6); // equivalent to exponent<<6
}

inline unsigned median(unsigned i,unsigned j) {
    return i+(j-1-i)/2;
}

inline void storeSIMDblock(int32_t v[],unsigned k,LeafEntry l[],unsigned i,unsigned j) {
    unsigned m=median(i,j);
    v[k+0]=l[m].key;
    v[k+1]=l[median(i,m)].key;
    v[k+2]=l[median(1+m,j)].key;
}

inline unsigned storeCachelineBlock(int32_t v[],unsigned k,LeafEntry l[],unsigned i,unsigned j) {
    storeSIMDblock(v,k+3*0,l,i,j);
    unsigned m=median(i,j);
    storeSIMDblock(v,k+3*1,l,i,median(i,m));
    storeSIMDblock(v,k+3*2,l,median(i,m)+1,m);
    storeSIMDblock(v,k+3*3,l,m+1,median(m+1,j));
    storeSIMDblock(v,k+3*4,l,median(m+1,j)+1,j);
    return k+16;
}

unsigned storeFASTpage(int32_t v[],unsigned offset,LeafEntry l[],unsigned i,unsigned j,unsigned levels) {
    for (unsigned level=0;level<levels;level++) {
        unsigned chunk=(j-i)/pow16(level);
        for (unsigned cl=0;cl<pow16(level);cl++)
            offset=storeCachelineBlock(v,offset,l,i+cl*chunk,i+(cl+1)*chunk);
    }
    return offset;
}

int32_t* buildFAST(LeafEntry l[],unsigned len) {
    // create array of appropriate size
    unsigned n=0;
    for (unsigned i=0; i<K+4; i++)
        n+=pow16(i);
    n=n*64/4;
    int32_t* v=(int32_t*)malloc_huge(sizeof(int32_t)*n);

    // build FAST
    unsigned offset=storeFASTpage(v,0,l,0,len,4);
    unsigned chunk=len/(1<<16);
    for (unsigned i=0;i<(1<<16);i++)
        offset=storeFASTpage(v,offset,l,i*chunk,(i+1)*chunk,K);
    assert(offset==n);

    return v;
}

inline void storeCudaBlock(int32_t v[], unsigned k, LeafEntry l[], unsigned i, unsigned j) {
    // calculate the 15 keys we need to do the comparison for a 4-level binary tree
    unsigned m1 = median(i, j);
    unsigned m2 = median(i, m1);
    unsigned m3 = median(m1 + 1, j);
    unsigned m4 = median(i, m2);
    unsigned m5 = median(m2 + 1, m1);
    unsigned m6 = median(m1 + 1, m3);
    unsigned m7 = median(m3 + 1, j);
    unsigned m8 = median(i, m4);
    unsigned m9 = median(m4 + 1, m2);
    unsigned m10 = median(m2 + 1, m5);
    unsigned m11 = median(m5 + 1, m1);
    unsigned m12 = median(m1 + 1, m6);
    unsigned m13 = median(m6 + 1, m3);
    unsigned m14 = median(m3 + 1, m7);
    unsigned m15 = median(m7 + 1, j);

    // store them
    v[k + 0] = l[m1].key;
    v[k + 1] = l[m2].key;
    v[k + 2] = l[m3].key;
    v[k + 3] = l[m4].key;
    v[k + 4] = l[m5].key;
    v[k + 5] = l[m6].key;
    v[k + 6] = l[m7].key;
    v[k + 7] = l[m8].key;
    v[k + 8] = l[m9].key;
    v[k + 9] = l[m10].key;
    v[k + 10] = l[m11].key;
    v[k + 11] = l[m12].key;
    v[k + 12] = l[m13].key;
    v[k + 13] = l[m14].key;
    v[k + 14] = l[m15].key;
}
//inline void storeCudaBlock(int32_t v[], unsigned k, LeafEntry l[], unsigned i, unsigned j) {
//    // calculate the 7 keys we need to do the comparison
//    unsigned m1 = median(i, j);
//    unsigned m2 = median(i, m1);
//    unsigned m3 = median(m1 + 1, j);
//    unsigned m4 = median(i, m2);
//    unsigned m5 = median(m2 + 1, m1);
//    unsigned m6 = median(m1 + 1, m3);
//    unsigned m7 = median(m3 + 1, j);
//
//    // store them
//    v[k + 0] = l[m1].key;
//    v[k + 1] = l[m2].key;
//    v[k + 2] = l[m3].key;
//    v[k + 3] = l[m4].key;
//    v[k + 4] = l[m5].key;
//    v[k + 5] = l[m6].key;
//    v[k + 6] = l[m7].key;
//}

inline unsigned storeCachelineBlockCuda(int32_t v[], unsigned k, LeafEntry l[], unsigned i, unsigned j) {
    // Store the root node in the first SIMD block
    storeCudaBlock(v, k, l, i, j);
    k += 16;

    // Return the updated offset, now pointing past the blocks for the second level
    return k;  // The offset is adjusted for the 9 SIMD blocks stored (1 root + 8 children)
}


unsigned storeCudaFastPage(int32_t v[], unsigned offset, LeafEntry l[], unsigned i, unsigned j, unsigned levels) {
    for (unsigned level=0;level<levels;level++) {
        unsigned chunk=(j-i)/pow16(level);
        for (unsigned cl=0;cl<pow16(level);cl++)
            offset= storeCachelineBlockCuda(v, offset, l, i + cl * chunk, i + (cl + 1) * chunk);
    }
    return offset;
}

int32_t* buildFASTCuda(LeafEntry l[], unsigned len) {
    // create array of appropriate size
    unsigned n=0;
    for (unsigned i=0; i<K+4; i++)
        n+=pow16(i);
    n*=16;
    int32_t* v=(int32_t*)malloc_huge(sizeof(int32_t)*n);

    // build FAST

    unsigned offset= storeCudaFastPage(v, 0, l, 0, len, 4);
    unsigned chunk = len/(1<<16);
    for (unsigned i=0;i<(1<<16);i++)
        offset= storeCudaFastPage(v, offset, l, i * chunk, (i + 1) * chunk, K);
    assert(offset==n);

    return v;
}

inline unsigned maskToIndex(unsigned bitmask) {
    static unsigned table[8]={0,9,1,2,9,9,9,3};
    return table[bitmask&7];
}

unsigned scale=0;

unsigned search(const int32_t v[],int32_t key_q) {
    __m128i xmm_key_q=_mm_set1_epi32(key_q);

    unsigned page_offset=0;
    unsigned level_offset=0;

    // first page
    for (unsigned cl_level=1; cl_level<=4; cl_level++) {
        // first SIMD block
        __m128i xmm_tree=_mm_loadu_si128((__m128i*) (v+page_offset+level_offset*16));
        __m128i xmm_mask=_mm_cmpgt_epi32(xmm_key_q,xmm_tree);
        unsigned index=_mm_movemask_ps(_mm_castsi128_ps(xmm_mask));
        unsigned child_index=maskToIndex(index);

        // second SIMD block
        xmm_tree=_mm_loadu_si128((__m128i*) (v+page_offset+level_offset*16+3+3*child_index));
        xmm_mask=_mm_cmpgt_epi32(xmm_key_q,xmm_tree);
        index=_mm_movemask_ps(_mm_castsi128_ps(xmm_mask));

        unsigned cache_offset=child_index*4 + maskToIndex(index);
        level_offset=level_offset*16 + cache_offset;
        page_offset+=pow16(cl_level);
    }

    unsigned pos=level_offset;
    unsigned offset=69904+level_offset*scale;
    page_offset=0;
    level_offset=0;

    // second page
    for (unsigned cl_level=1; cl_level<=K; cl_level++) {
        // first SIMD block
        __m128i xmm_tree=_mm_loadu_si128((__m128i*) (v+offset+page_offset+level_offset*16));
        __m128i xmm_mask=_mm_cmpgt_epi32(xmm_key_q,xmm_tree);
        unsigned index=_mm_movemask_ps(_mm_castsi128_ps(xmm_mask));
        unsigned child_index=maskToIndex(index);

        // second SIMD block
        xmm_tree=_mm_loadu_si128((__m128i*) (v+offset+page_offset+level_offset*16+3+3*child_index));
        xmm_mask=_mm_cmpgt_epi32(xmm_key_q,xmm_tree);
        index=_mm_movemask_ps(_mm_castsi128_ps(xmm_mask));

        unsigned cache_offset=child_index*4 + maskToIndex(index);
        level_offset=level_offset*16 + cache_offset;
        page_offset+=pow16(cl_level);
    }

    return (pos<<(K*4))|level_offset;
}


#define BUF_SIZE 2048

#define THREAD_NUM 256

std::vector<int> read_data(const char *path) {
    std::vector<int> vec;
    FILE *fin = fopen(path, "rb");
    int buf[BUF_SIZE];
    while (true) {
        size_t num_read = fread(buf, sizeof(int), BUF_SIZE, fin);
        for (int i = 0; i < num_read; i++) {
            vec.push_back(buf[i]);
        }
        if (num_read < BUF_SIZE) break;
    }
    fclose(fin);
    return vec;
}
#define QUERIES_PER_TRIAL (50 * 1000 * 1000)

std::atomic<unsigned long long> check_atomic(0);  // 使用原子变量以避免并发问题

void threadedSearch(const int32_t* v, const std::vector<int>& queries, unsigned i) {
    for(int j=0;j<QUERIES_PER_TRIAL/THREAD_NUM;++j) {
        int k = j + i*((QUERIES_PER_TRIAL+THREAD_NUM-1)/THREAD_NUM);
        if (k>=QUERIES_PER_TRIAL)
            return;
        unsigned result = search(v, queries[k]);
        check_atomic += result;
    }
    // 安全地更新全局计数
}

void parallelSearch(const int32_t* v, const std::vector<int>& queries, unsigned scale) {
    boost::asio::thread_pool pool(THREAD_NUM);
    std::vector<std::thread> threads;
    for (int i=0; i<THREAD_NUM;++i) {
        threads.emplace_back(threadedSearch, v, queries, i);
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}



extern int32_t * pinCuda(int32_t *fast, unsigned n);

extern uint64_t cudaSearch(std::vector<int>& queries, const int32_t* fast, unsigned scale, int* res);

int main(int argc,char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " DATA_PATH" << std::endl;
        exit(1);
    }

    std::vector<int> keys = read_data(argv[1]);
    printf("num elements: %lu\n", keys.size());
    // Clone vec so we don't bring pages from it into cache when selecting random keys

    std::vector<int> keys_clone(keys.begin(), keys.end());

    keys.push_back(INT_MAX);
    unsigned n=(1<<(16+(K*4))); // note: padding to power of 2 for FAST
    for (unsigned i = keys.size(); i < n; i++) {
        keys.push_back(INT_MAX);
    }
    printf("num elements (padded): %d\n", n);
    auto build_start = clock();
    LeafEntry* leaves = new LeafEntry[n];
    for (unsigned i = 0; i < n; i++) {
        leaves[i].key = keys[i];
        leaves[i].value = i;
    }

    int32_t *fastCuda = buildFASTCuda(leaves, n);
    auto build_end = clock();
    printf("FAST build time taken: %lf ns\n",
           double(build_end - build_start) / CLOCKS_PER_SEC * 1e9);
    int32_t * cudaMem = pinCuda(fastCuda, n);
    build_end = clock();

    printf("FAST build time taken: %lf ns (including pinning in cuda memory)\n",
           double(build_end - build_start) / CLOCKS_PER_SEC * 1e9);
    for (unsigned i = 0; i < K; i++)
        scale += pow16(i);
    scale *= 16;

    uint32_t seed = std::random_device()();
    std::mt19937 rng;
    std::uniform_int_distribution<> dist(0, keys_clone.size()-1);
    std::vector<int> queries(QUERIES_PER_TRIAL);

    rng.seed(seed);
    for (int &query : queries) {
        query = keys_clone[dist(rng)];
    }
    int* checks = new int[QUERIES_PER_TRIAL];
    unsigned long long int check = 0;
    auto start = std::chrono::high_resolution_clock::now();
//    for (const int& key : queries) {
//        check += search(fastCuda, key);
//    }
    cudaSearch(queries, cudaMem, scale, checks);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> elapsed_cuda = end - start;
    for(int i=0;i<QUERIES_PER_TRIAL;++i) {
        check += checks[i];
    }
    printf("FAST average time taken: %lf ns\n",
           elapsed_cuda/ queries.size() );
    printf("FAST checksum (can be different from other range search baselines) = %ld\n", check);

    delete[] checks;

    int32_t *fast = buildFAST(leaves, n);
//    check = 0;
    auto start_multi = std::chrono::high_resolution_clock::now();
    parallelSearch(fast, queries, scale);
    auto end_multi = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::nano> elapsed = end_multi - start_multi;
    printf("FAST average time taken: %lf ns\n",
           elapsed.count()/queries.size());
    printf("FAST checksum (can be different from other range search baselines) = %llu\n", check_atomic.load());

    return 0;
}
