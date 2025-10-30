#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* arr_in,
    const unsigned int* hist_pref /* size == gridDim.x * RADIX_BINS */,
    const unsigned int* hist_pref_pref /* size == gridDim.x * RADIX_BINS */,
    unsigned int* arr_out,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int block_local[GROUP_SIZE];

    if (idx < n) {
        block_local[threadIdx.x] = arr_in[idx];
    }
    __syncthreads();

    if (idx < n) {
        const unsigned int elem = arr_in[idx];
        const unsigned int elem_bin = ELEM_BIN(elem, offset);
        unsigned int elem_out_idx = 0;

        // How many numbers globally in array are smaller than `elem`?
        if (elem_bin > 0) {
            elem_out_idx += hist_pref_pref[(gridDim.x - 1) * RADIX_BINS + elem_bin - 1];
        }

        // How many numbers in previous blocks array are equal to `elem` but have smaller index?
        if (blockIdx.x > 0) {
            elem_out_idx += hist_pref[(blockIdx.x - 1) * RADIX_BINS + elem_bin];
        }

        // How many numbers in current block are equal to `elem` but have smaller index?
#pragma unroll
        for (unsigned int i = 0; i < threadIdx.x; i++) {
            elem_out_idx += (ELEM_BIN(block_local[i], offset) == elem_bin);
        }

        arr_out[elem_out_idx] = elem;
    }
}

namespace cuda {
void radix_sort_04_scatter(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& arr_in,
    const gpu::gpu_mem_32u& hist_pref,
    const gpu::gpu_mem_32u& hist_pref_pref,
    gpu::gpu_mem_32u& arr_out,
    unsigned int n,
    unsigned int offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(arr_in.cuptr(), hist_pref.cuptr(), hist_pref_pref.cuptr(), arr_out.cuptr(), n, offset);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda