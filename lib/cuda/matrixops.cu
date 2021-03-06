// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Lijun Zhu
// california institute of technology
// (c) 2016-2019  all rights reserved
//

// my declaration
#include "matrixops.h"
// cuda utitlies
#include "cudalib.h"


namespace matrixops_kernels {
    template<typename T>
    __global__ void _identity(T * const m, const size_t n);

    template<typename Tout, typename Tin>
    __global__ void _copy_tile(Tout * const odata,  const size_t ldo,
                const Tin * const idata, const size_t ldi, const size_t rows, const size_t cols);

    template<typename T>
    __global__ void _copy_indices(T * const odata,  const size_t ldo,
                const T * const idata, const size_t ldi, const size_t rows, const size_t cols, const size_t * const indices);

    template<typename T>
    __global__ void _transpose(T* const odata, const T* const idata, const size_t nrows, const size_t ncols);

    template<typename T>
    __global__ void _duplicate_vector(T * const odata,  const size_t ldo,
                const T * const idata, const size_t incx, const size_t m, const size_t n);

    template<typename T>
    __global__ void _copy_triangle(T * const gdata, const size_t n, const int fill);

} // of namespace matrixops_kernels

// identity matrix
template<typename T>
void cudalib::matrix::
identity(T * const mat, const size_t n, cudaStream_t stream)
{
    //set matrix to 0 at first
    cudaSafeCall(cudaMemsetAsync(mat, 0, n*n*sizeof(T), stream));
    dim3 blockSize(BLOCKDIM);
    dim3 gridSize(IDIVUP(n, blockSize.x));
    matrixops_kernels::_identity<T><<<gridSize, blockSize, 0, stream>>>(mat, n);
    cudaCheckError("matrixops_kernels::_identidy");
}

// explicit instantiation
template void cudalib::matrix::identity<float>(float * const, const size_t, cudaStream_t);
template void cudalib::matrix::identity<double>(double * const, const size_t, cudaStream_t);

// copy a tile of data (submatrix or patch)
template<typename Tout, typename Tin>
void cudalib::matrix::
copy_tile(Tout* const odata,  const size_t ldo,
                const size_t o_m_start, const size_t o_n_start, // starting position of odata
                const Tin* const idata, const size_t ldi,
                const size_t i_m_start, const size_t i_n_start, // starting position of idata
                const size_t m, const size_t n, // tile to be copied
                cudaStream_t stream)
{
    // move input data pointer to the starting corner
    const Tin * idata_start = idata + i_m_start*ldi + i_n_start;
    Tout * odata_start = odata + o_m_start*ldo + o_n_start;
    dim3 blockSize(16, 16, 1); // typically (16x16x1)
    dim3 gridSize(IDIVUP(n, blockSize.x), IDIVUP(m, blockSize.y), 1);
    matrixops_kernels::_copy_tile<Tout, Tin><<<gridSize, blockSize, 0, stream>>>
        (odata_start, ldo, idata_start,ldi, m, n);
    cudaCheckError("matrixops_kernels::_copy_tile");
}

// explicit instantiation
template void cudalib::matrix::copy_tile<float, float>(float * const, const size_t, const size_t, const size_t,
    const float * const, const size_t, const size_t, const size_t, const size_t, const size_t, cudaStream_t);
template void cudalib::matrix::copy_tile<double, double>(double * const, const size_t, const size_t, const size_t,
    const double * const, const size_t, const size_t, const size_t, const size_t, const size_t, cudaStream_t);
template void cudalib::matrix::copy_tile<float, double>(float * const, const size_t, const size_t, const size_t,
    const double * const, const size_t, const size_t, const size_t, const size_t, const size_t, cudaStream_t);
template void cudalib::matrix::copy_tile<double, float>(double * const, const size_t, const size_t, const size_t,
    const float * const, const size_t, const size_t, const size_t, const size_t, const size_t, cudaStream_t);



// copy data according to a list of indices
// i.e., odata[x,y] = idata[x, indices[y]]
template<typename T>
void cudalib::matrix::
copy_indices(T * const odata,  const size_t ldo,
                const T * const idata, const size_t ldi,
                const size_t m, const size_t n, // tile to be copied
                const size_t * const indices, // indices for cols to be copied
                cudaStream_t stream)
{
    dim3 blockSize(BDIMX, BDIMY); // typically (16x16x1)
    dim3 gridSize(IDIVUP(n, blockSize.x), IDIVUP(m, blockSize.y));
    matrixops_kernels::_copy_indices<T><<<gridSize, blockSize, 0, stream>>>
        (odata, ldo, idata, ldi, m, n, indices);
    cudaCheckError("matrixops_kernels::_copy_indices");
}

// explicit instantiation
template void cudalib::matrix::copy_indices<float>(float * const, const size_t,
    const float * const, const size_t, const size_t, const size_t, const size_t * const, cudaStream_t);
template void cudalib::matrix::copy_indices<double>(double * const, const size_t,
    const double * const, const size_t, const size_t, const size_t, const size_t * const, cudaStream_t);



// matrix transpose with shared mem, unrolling and memory padding algorithm
// idata a matrix (iM x iN) with leading dimension iN
// odata a transposed matrix (iN x iM) with leading dimension iM
template<typename T>
void cudalib::matrix::
transpose(T * const odata, const T* const idata, const size_t iM, const size_t iN, cudaStream_t stream)
{
    dim3 blockSize(BDIMX, BDIMY);
    // use blockX for leading dimension
    dim3 gridSize(IDIVUP(iN, blockSize.x), IDIVUP(iM,blockSize.y));
    // divide gridsize for unrolling (two lines per thread)T* const odata, const T* const idata, const size_t nrows, const size_t ncols)
    dim3 gridSize2(IDIVUP(gridSize.x, 2), gridSize.y);

    matrixops_kernels::_transpose<T><<<gridSize2, blockSize, 0, stream>>> (
        odata, idata, iM, iN);
    cudaCheckError("matrixops_kernels::_transpose kernel error");
}

// explicit specialization
template void cudalib::matrix::transpose<float>(float * const, const float *, const size_t, const size_t, cudaStream_t);
template void cudalib::matrix::transpose<double>(double * const, const double *, const size_t, const size_t, cudaStream_t);
template void cudalib::matrix::transpose<int>(int * const, const int *, const size_t, const size_t, cudaStream_t);



// duplicate a vector to multiple rows of a matrix
template<typename T>
void cudalib::matrix::
duplicate_vector(T* const odata,  const size_t ldo,
                const T* const idata, const size_t incx,
                const size_t m, const size_t n, // tile to be copied
                cudaStream_t stream)
{
    // move input data pointer to the starting corner
    dim3 blockSize(16, 16, 1); // typically (16x16x1)
    dim3 gridSize(IDIVUP(n, blockSize.x), IDIVUP(m, blockSize.y), 1);
    matrixops_kernels::_duplicate_vector<T><<<gridSize, blockSize, 0, stream>>>
        (odata, ldo, idata, incx, m, n);
    cudaCheckError("matrixops_kernels::_duplicate_vector");
}

// explicit instantiation for shared library
template void cudalib::matrix::duplicate_vector<float>(float * const, const size_t,
    const float * const, const size_t, const size_t, const size_t, cudaStream_t);
template void cudalib::matrix::duplicate_vector<double>(double * const, const size_t,
    const double * const, const size_t, const size_t, const size_t, cudaStream_t);


// duplicate the upper triangle to lower or vice versa for a nxn matrix
// fill = 0, the lower is filled, copy to upper
// fill = 1, the upper is filled, copy to lower
template<typename T>
void cudalib::matrix::
copy_triangle(T* const gdata, const size_t n, const int fill, cudaStream_t stream)
{
    int blockSize = BLOCKDIM;
    int elements = n*(n-1)/2;
    int gridSize = IDIVUP(elements, blockSize);
    matrixops_kernels::_copy_triangle<T><<<gridSize, blockSize, 0, stream>>>
        (gdata, n, fill);
    cudaCheckError("matrixops_kernels::_copy_triangle");
}

// explicit instantiation for shared library
template void cudalib::matrix::copy_triangle<float>(float * const, const size_t, const int, cudaStream_t);
template void cudalib::matrix::copy_triangle<double>(double * const, const size_t, const int, cudaStream_t);


//************** CUDA KERNELS **************
// set diagonal elements for identity matrix
template<typename T>
__global__ void
matrixops_kernels::_identity(T * const mat,  const size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        mat[IDX2R(i, i, n)] = (T)1.0f;
}

// To achieve high bandwidth, use x for leading dimension (col) to match the thread block major
// copy rows x cols from idata to odata
template<typename Tout, typename Tin>
__global__ void
matrixops_kernels::_copy_tile(Tout * const odata,  const size_t ldo,
                const Tin * const idata, const size_t ldi, const size_t m, const size_t n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < m)
        odata[IDX2R(y, x, ldo)] = (Tout)idata[IDX2R(y, x, ldi)];
}

// To achieve high bandwidth, use x for leading dimension (col) to match the thread block major
// copy rows x cols from idata to odata
//
template<typename T>
__global__ void
matrixops_kernels::_copy_indices(T * const odata,  const size_t ldo,
                const T * const idata, const size_t ldi, const size_t m, const size_t n, const size_t * const indices)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < m) {
        int index = indices[x];
        odata[y*ldo + x] = idata[y*ldi+index];
    }
}

// kernel
template<typename T>
__global__ void matrixops_kernels::
_transpose(T* const odata, const T* const idata, const size_t nrows, const size_t ncols)
{
    // use a tile of shared memory as transpose buffer
    __shared__ float tile[BDIMY][BDIMX * 2 + 2];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = IDX2R(row, col, ncols);
    unsigned int offset2 = IDX2R(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = IDX2C(row, col, nrows);
    unsigned int transposed_offset2 = IDX2C(row2, col2, nrows);

    if (row < nrows && col < ncols)
    {
        tile[threadIdx.y][threadIdx.x] = idata[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = idata[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols)
    {
        odata[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols)
    {
        odata[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}

// To achieve high bandwidth, use x for leading dimension (col) to match the thread block major
// copy rows x cols from idata to odata
//
template<typename T>
__global__ void
matrixops_kernels::_duplicate_vector(T * const odata,  const size_t ldo,
                const T * const idata, const size_t incx, const size_t m, const size_t n)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < m)
        odata[y*ldo + x] = idata[incx*x];
}

template<typename T>
__global__ void
matrixops_kernels::_copy_triangle(T * const gdata, const size_t n, const int fill)
{
    // get thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n*(n-1)/2) return;

    // find the row, col indices
    bool found = 0;
    int row = 1;
    while(!found) {
        if (tid >= row*(row-1)/2 && tid < row*(row+1)/2) found = 1;
        else row++;
    }
    int col = tid - row*(row-1)/2;

    if (fill == 0) //lower is filled, copy to upper
        gdata[IDX2R(col, row, n)] = gdata[IDX2R(row, col, n)];
    else // upper is filled, copy to lower
        gdata[IDX2R(row, col, n)] = gdata[IDX2R(col, row, n)];
    // all done
}

// end of file
