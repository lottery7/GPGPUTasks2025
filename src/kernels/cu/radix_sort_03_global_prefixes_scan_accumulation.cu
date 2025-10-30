#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* pow2_sum,
    unsigned int* prefix_sum_accum,
    unsigned int pow2)
{
    if ((blockIdx.x >> pow2) & 0x1) {
#pragma unroll
        for (unsigned int i = threadIdx.x; i < RADIX_BINS; i += blockDim.x) {
            // const unsigned int pow2_idx = blockIdx.x / (1 << pow2) - 1;
            const unsigned int pow2_idx = (blockIdx.x >> pow2) - 1;
            prefix_sum_accum[blockIdx.x * RADIX_BINS + i] += pow2_sum[pow2_idx * RADIX_BINS + i];
        }
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum,
    gpu::gpu_mem_32u& prefix_sum_accum,
    unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda