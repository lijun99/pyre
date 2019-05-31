// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

/// @file reduction.h
/// @brief reduction functions within a block

#ifndef cudalib_reduction_h
#define cudalib_reduction_h

#include <cooperative_groups.h>

namespace cg = ::cooperative_groups;

namespace reduction_kernels {

template <typename T>
__device__ T sum_reduce_block(T *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    auto beta  = sdata[tid];

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
            beta += tile32.shfl_down(beta, i);
    }
    sdata[tid] = beta;
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        beta  = 0;
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            beta  += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
    return beta;
}

template <typename T>
__device__ T product_reduce_block(T *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    auto beta  = sdata[tid];

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
            beta *= tile32.shfl_down(beta, i);
    }
    sdata[tid] = beta;
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        beta  = 0;
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            beta  *= sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
    return beta;
}

template <typename T>
__device__ T max_reduce_block(T *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    auto beta  = sdata[tid];

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
            beta = max(beta, tile32.shfl_down(beta, i));
    }
    sdata[tid] = beta;
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            beta  = max(beta, sdata[i]);
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
    return beta;
}

template <typename T>
__device__ T min_reduce_block(T *sdata, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    auto beta  = sdata[tid];

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
            beta = min(beta, tile32.shfl_down(beta, i));
    }
    sdata[tid] = beta;
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            beta  = min(beta, sdata[i]);
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
    return beta;
}

} // of namespace reduction_kernels

#endif // cudalib_reduction_h
// end of file

