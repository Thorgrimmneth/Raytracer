enum ObjectType{
    SPHERE,
    TRIANGLE,
    PLANE,
    IMPLICIT
};

struct BaseObject{
    float3 position;
    ObjectType type;
};