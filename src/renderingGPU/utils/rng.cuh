#pragma once

#include "simplifiedDef.cuh"

struct RNG
{
    uint state;

    DEVICE 
    RNG(uint seed)
        : state(seed)
    {}

    D_FORCEINLINE 
    uint nextUInt()
    {
        uint oldstate = state;

        state = oldstate * 747796405u + 2891336453u;

        uint word =
            ((oldstate >> ((oldstate >> 28u) + 4u))
            ^ oldstate)
            * 277803737u;

        return (word >> 22u) ^ word;
    }

    D_FORCEINLINE  
    float nextFloat()
    {
        return (nextUInt() >> 8) * 0x1p-24f;
    }
};

D_FORCEINLINE 
uint pcg_hash(uint input)
{
    uint state = input * 747796405u + 2891336453u;

    uint word = ((state >> ((state >> 28u) + 4u)) ^ state)
                * 277803737u;

    return (word >> 22u) ^ word;
}