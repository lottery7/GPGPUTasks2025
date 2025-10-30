#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* pow2_sum /* contains (2 * gridDim.x) * RADIX_BINS values */,
    unsigned int* next_pow2_sum /* will contain gridDim.x * RADIX_BINS values */)
{
#pragma unroll
    for (unsigned int i = threadIdx.x; i < RADIX_BINS; i += blockDim.x) {
        unsigned int acc = 0;
        if (2 * blockIdx.x < 2 * gridDim.x) {
            acc += pow2_sum[(2 * blockIdx.x) * RADIX_BINS + i];
        }
        if (2 * blockIdx.x + 1 < 2 * gridDim.x) {
            acc += pow2_sum[(2 * blockIdx.x + 1) * RADIX_BINS + i];
        }
        next_pow2_sum[blockIdx.x * RADIX_BINS + i] = acc;
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& pow2_sum,
    const gpu::gpu_mem_32u& next_pow2_sum)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda