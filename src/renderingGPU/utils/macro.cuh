#pragma once

#ifndef HD_INLINE
#define HD_INLINE __host__ __device__ inline
#endif

#ifndef D_FORCEINLINE
#define D_FORCEINLINE __device__ __forceinline__
#endif

#ifndef H_INLINE
#define H_INLINE __host__ inline
#endif

#ifndef HD_FORCEINLINE
#define HD_FORCEINLINE __host__ __device__ __forceinline__
#endif

#ifndef HD
#define HD __host__ __device__
#endif

#ifndef HOST
#define HOST __host__
#endif

#ifndef DEVICE
#define DEVICE __device__
#endif

#ifndef GLOBAL
#define GLOBAL __global__
#endif