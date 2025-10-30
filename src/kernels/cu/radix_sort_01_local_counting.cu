#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* arr_in,
    unsigned int* hist /* size == gridDim.x * RADIX_BINS */,
    unsigned int* hist_pref /* size == gridDim.x * RADIX_BINS */,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int local_hist[RADIX_BINS];
#pragma unroll
    for (unsigned int i = threadIdx.x; i < RADIX_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    if (idx < n) {
        unsigned int elem_bin = (arr_in[idx] >> offset) & RADIX_MASK;
        atomicAdd(&local_hist[elem_bin], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned int i = 1; i < RADIX_BINS; i++) {
            local_hist[i] += local_hist[i - 1];
        }
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = threadIdx.x; i < RADIX_BINS; i += blockDim.x) {
        hist[blockIdx.x * RADIX_BINS + i] = local_hist[i] - (i > 0 ? local_hist[i - 1] : 0);
        hist_pref[blockIdx.x * RADIX_BINS + i] = local_hist[i];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& arr_in,
    gpu::gpu_mem_32u& hist,
    gpu::gpu_mem_32u& hist_pref,
    unsigned int n,
    unsigned int offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        arr_in.cuptr(), hist.cuptr(), hist_pref.cuptr(), n, offset);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda