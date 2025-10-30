#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RADIX_BITS 8
#define RADIX_BINS (1 << RADIX_BITS)
#define RADIX_MASK (RADIX_BINS - 1)
#define ELEM_BIN(elem, offset) ((elem >> offset) & RADIX_MASK)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
