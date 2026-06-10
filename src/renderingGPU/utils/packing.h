#pragma once
#include <cstdint>

static __forceinline__ __device__
void packPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1
)
{
    const uint64_t uptr =
        reinterpret_cast<uint64_t>(ptr);

    i0 = uptr >> 32;
    i1 = uptr & 0xFFFFFFFF;
}

static __forceinline__ __device__
void* unpackPointer(
    uint32_t i0,
    uint32_t i1
)
{
    const uint64_t uptr =
        (static_cast<uint64_t>(i0) << 32) |
        static_cast<uint64_t>(i1);

    return reinterpret_cast<void*>(uptr);
}