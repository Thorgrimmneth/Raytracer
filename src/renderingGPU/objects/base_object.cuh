#include "../objectsUtils/aabb.cuh"

enum ObjectType{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT
};

struct BaseObject{
    AABB bbox;
    ObjectType type;
    int index;
};