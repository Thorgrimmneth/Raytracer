#include "../objectsUtils/aabb.cuh"
#include <stdint.h>

enum ObjectType : uint32_t{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT
};

struct DataMin{
    float3 min;
    ObjectType type;
};

struct DataMax{
    float3 max;
    int index;
};

struct BaseObject{
    DataMin min;
    DataMax max;

    BaseObject(float3 min, float3 max, ObjectType type, int index) : min(DataMin{min, type}), max(DataMax{max, index}) {}

    __host__ __device__
    float3 getMin() const{
        return min.min;
    };

    __host__ __device__
    float3 getMax() const{
        return max.max;
    };

    __device__
    ObjectType getType() const{
        return min.type;
    };

    __device__
    int getIndex() const{
        return max.index;
    };
};