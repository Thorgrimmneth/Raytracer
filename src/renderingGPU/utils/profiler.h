#pragma once

#include <nvtx3/nvToolsExt.h>

namespace Profiler
{
    class ScopedRange
    {
    public:
        explicit ScopedRange(const char* name)
        {
            nvtxRangePushA(name);
        }

        ~ScopedRange()
        {
            nvtxRangePop();
        }
    };
}

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define PROFILE_SCOPE(name) \
    Profiler::ScopedRange CONCAT(_nvtx_scope_, __LINE__)(name)