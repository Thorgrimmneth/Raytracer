#pragma once

struct RNG
{
    uint state;

    __device__ RNG(uint seed)
        : state(seed)
    {}

    __device__ uint nextUInt()
    {
        uint oldstate = state;

        state = oldstate * 747796405u + 2891336453u;

        uint word =
            ((oldstate >> ((oldstate >> 28u) + 4u))
            ^ oldstate)
            * 277803737u;

        return (word >> 22u) ^ word;
    }

    __device__ float nextFloat()
    {
        return (nextUInt() >> 8) * 0x1p-24f;
    }
};

__device__ inline
uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;

    uint word = ((state >> ((state >> 28u) + 4u)) ^ state)
                * 277803737u;

    return (word >> 22u) ^ word;
}