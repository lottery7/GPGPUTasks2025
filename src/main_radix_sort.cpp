#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>

// #define PRINT_GPU_ARRAY_ON

void print_gpu_array(const std::string& message, const gpu::gpu_mem_32u& arr, int bits_count = 0)
{
#ifndef PRINT_GPU_ARRAY_ON
    return;
#endif
    std::cout << message;
    auto arr_cpu = arr.readVector();
    for (size_t i = 0; i < arr_cpu.size(); ++i) {
        if (bits_count > 0) {
            unsigned int x = arr_cpu[i];
            for (int bit = bits_count - 1; bit >= 0; --bit) {
                std::cout << ((x >> bit) & 1);
            }
        } else {
            std::cout << arr_cpu[i];
        }
        std::cout << ((i + 1) % RADIX_BINS == 0 ? '|' : ' ');
    }
    std::cout << std::endl;
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    unsigned int n = 100 * 1000 * 1000;
    int max_value = std::numeric_limits<int>::max();
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

    {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << n / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::WorkSize work_size(GROUP_SIZE, n);
    const unsigned int grid_size = work_size.cuGridSize().x;

    gpu::gpu_mem_32u input_gpu(n);

    gpu::gpu_mem_32u hist_gpu(grid_size * RADIX_BINS);
    gpu::gpu_mem_32u hist_pref_gpu(grid_size * RADIX_BINS);

    gpu::gpu_mem_32u hist_pref_sum_accum_gpu(grid_size * RADIX_BINS);
    gpu::gpu_mem_32u hist_pref_pref_gpu(grid_size * RADIX_BINS);
    gpu::gpu_mem_32u pow2_sum_buffer1_gpu(grid_size * RADIX_BINS);
    gpu::gpu_mem_32u pow2_sum_buffer2_gpu(grid_size * RADIX_BINS);

    gpu::gpu_mem_32u sort_buffer1_gpu(n);
    gpu::gpu_mem_32u sort_buffer2_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    print_gpu_array("initial: ", input_gpu, 5);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        input_gpu.copyToN(sort_buffer1_gpu, input_gpu.number());

        auto &sort_input = sort_buffer1_gpu, sort_output = sort_buffer2_gpu;
        bool swap_output = true;
        for (int offset = 0; (1 << offset) <= max_value && offset < 32; offset += RADIX_BITS) {
            cuda::fill_buffer_with_zeros(work_size, hist_gpu, hist_gpu.number());
            cuda::radix_sort_01_local_counting(work_size, sort_input, hist_gpu, hist_pref_gpu, n, offset);

            print_gpu_array("hist: ", hist_gpu);
            print_gpu_array("hist_pref in block: ", hist_pref_gpu);

            hist_pref_gpu.copyToN(hist_pref_pref_gpu, hist_gpu.number());
            hist_pref_gpu.copyToN(pow2_sum_buffer1_gpu, hist_gpu.number());

            auto &pow2_sum_in = pow2_sum_buffer1_gpu, &pow2_sum_out = pow2_sum_buffer2_gpu;
            for (unsigned int pow2 = 0; (1 << pow2) < grid_size; pow2++) {
                if (pow2 > 0) {
                    auto num_blocks = div_ceil<unsigned int>(grid_size, 1 << pow2);
                    gpu::WorkSize reduction_work_size(GROUP_SIZE, num_blocks * GROUP_SIZE);
                    cuda::radix_sort_02_global_prefixes_scan_sum_reduction(reduction_work_size, pow2_sum_in, pow2_sum_out);
                    std::swap(pow2_sum_in, pow2_sum_out);
                }
                cuda::radix_sort_03_global_prefixes_scan_accumulation(work_size, pow2_sum_in, hist_pref_pref_gpu, pow2);
            }

            hist_gpu.copyToN(hist_pref_sum_accum_gpu, hist_gpu.number());
            hist_gpu.copyToN(pow2_sum_buffer1_gpu, hist_gpu.number());

            pow2_sum_in = pow2_sum_buffer1_gpu;
            pow2_sum_out = pow2_sum_buffer2_gpu;
            for (unsigned int pow2 = 0; (1 << pow2) < grid_size; pow2++) {
                if (pow2 > 0) {
                    auto num_blocks = div_ceil<unsigned int>(grid_size, 1 << pow2);
                    gpu::WorkSize reduction_work_size(GROUP_SIZE, num_blocks * GROUP_SIZE);
                    cuda::radix_sort_02_global_prefixes_scan_sum_reduction(reduction_work_size, pow2_sum_in, pow2_sum_out);
                    std::swap(pow2_sum_in, pow2_sum_out);
                }
                cuda::radix_sort_03_global_prefixes_scan_accumulation(work_size, pow2_sum_in, hist_pref_sum_accum_gpu, pow2);
            }

            print_gpu_array("hist_pref sum: ", hist_pref_sum_accum_gpu);

            print_gpu_array("hist_pref_pref: ", hist_pref_pref_gpu);

            cuda::radix_sort_04_scatter(work_size, sort_input, hist_pref_sum_accum_gpu, hist_pref_pref_gpu, sort_output, n, offset);
            std::swap(sort_input, sort_output);
            swap_output = !swap_output;

            print_gpu_array("offset " + std::to_string(offset) + ": ", sort_input, 5);
        }

        if (swap_output) {
            std::swap(sort_buffer1_gpu, sort_buffer2_gpu);
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = sort_buffer2_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}