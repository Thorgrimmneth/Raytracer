#include "../scene/scene.cuh"
#include "light.cuh"

DEVICE 
LightSample Light::sampleCylinder(const float3 &p_point, RNG&rng) const
{
    float u = rng.nextFloat();
    float v = rng.nextFloat();

    float3 dir = normalize(getDirection());

    float3 uVec = normalize(cross(dir, make_float3(1.f, 0.f, 0.f)));

    if (length(uVec) < 1e-3f)
    {
        uVec = normalize(cross(dir, make_float3(0.f, 1.f, 0.f)));
    }

    float3 vVec = normalize(cross(dir, uVec));

    float z = u * getHeight();
    float phi = v * 2.f * GPUPIf;

    float3 pointOnCircle = getRadius() * (cosf(phi) * uVec + sinf(phi) * vVec);

    float3 randomPos = getPosition() + z * dir + pointOnCircle;

    float3 axisPoint = getPosition() + z * dir;

    float3 normal = normalize(randomPos - axisPoint);

    float3 lightDir = normalize(randomPos - p_point);

    float dist = distance(p_point, randomPos);

    float cosTheta = dot(normal, -lightDir);

    if (cosTheta <= 0.f)
    {
        LightSample s{};
        s.pdf = 0.f;
        return s;
    }

    float area = 2.f * GPUPIf * getRadius() * (getRadius() + getHeight());

    float pdf = (dist * dist) / (area * cosTheta);

    LightSample sample{};

    sample.radiance = getColorPower();
    sample.pdf = pdf;
    sample.power = getIntensity();
    sample.distance = dist;
    sample.direction = lightDir;

    return sample;
}

DEVICE 
LightSample Light::sampleDirectionnal(const float3 &p_point) const
{
    LightSample rep{};

    rep.radiance = getColorPower();
    rep.pdf = 1.f;
    rep.power = getIntensity();
    rep.distance = 1e20f;
    rep.direction = normalize(-getDirection());

    return rep;
}

DEVICE 
LightSample Light::samplePoint(const float3 &p_point) const
{
    float dist = distance(p_point, getPosition());

    float3 radiance = getColorPower() / (dist * dist);

    LightSample rep{};

    rep.radiance = radiance;
    rep.pdf = 1.f;
    rep.power = getIntensity();
    rep.distance = dist;
    rep.direction = normalize(getPosition() - p_point);

    return rep;
}

DEVICE 
LightSample Light::sampleQuad(const float3 &p_point, RNG&rng) const
{
    float3 randomPos = getPosition() + rng.nextFloat() * getDirection() + rng.nextFloat() * getV();

    float3 lightDir = normalize(randomPos - p_point);

    float dist = distance(p_point, randomPos);

    float cosTheta = dot(getNormal(), -lightDir);

    if (cosTheta <= 0.f)
    {
        LightSample s{};
        s.pdf = 0.f;
        return s;
    }

    float area = length(cross(getDirection(), getV()));

    LightSample rep{};

    rep.radiance = getColorPower();
    rep.pdf = (dist * dist) / (area * cosTheta);

    rep.power = getIntensity();
    rep.distance = dist;
    rep.direction = lightDir;

    return rep;
}

DEVICE 
LightSample Light::sampleCone(const float3 &p_point, RNG&rng) const
{
    float sunAngularRadius = 3.f * GPUPIf / 180.f;

    float u1 = rng.nextFloat();
    float u2 = rng.nextFloat();

    float cosTheta = 1.0f - u1 * (1.0f - cosf(sunAngularRadius));

    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    float phi = 2.0f * GPUPIf * u2;

    float3 w = normalize(getDirection());

    float3 up = fabs(w.y) < 0.99f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

    float3 u = normalize(cross(up, w));

    float3 v = cross(w, u);

    float3 sampledDir = normalize(u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta);

    float cosMax = cosf(sunAngularRadius);

    LightSample rep{};

    rep.direction = sampledDir;
    rep.distance = 1e20f;
    rep.radiance = getColorPower();

    rep.pdf = 1.0f / (2.0f * GPUPIf * (1.0f - cosMax));

    rep.power = getIntensity();

    return rep;
}

DEVICE 
LightSample Light::sampleSphereGeom(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    const Sphere &s = scene.spheres[getGeomIndex()];

    const Material &m = scene.materials[s.getMaterialIndex()];

    float z = 1.f - 2.f * rng.nextFloat();

    float r = sqrtf(max(0.f, 1.f - z * z));

    float phi = 2.f * GPUPIf * rng.nextFloat();

    float3 n = make_float3(r * cosf(phi), r * sinf(phi), z);

    float3 p = s.getCenter1() + s.getRadius() * n;

    float3 wi = normalize(p - p_point);

    float dist2 = length2(p - p_point);

    LightSample ls{};

    float cosThetaLight = max(dot(n, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float area = 4.f * GPUPIf * s.getRadius() * s.getRadius();

    float pdf_area = 1.f / area;

    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = sqrtf(dist2);
    ls.radiance = m.color() * m.intensity();

    ls.pdf = pdf;
    //ls.normal = n;

    return ls;
}

DEVICE 
LightSample Light::sampleImplicitSphereGeom(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    const ImplicitSphere &s = scene.implicitSpheres[getGeomIndex()];

    const Material &m = scene.materials[s.getMaterialIndex()];

    float z = 1.f - 2.f * rng.nextFloat();

    float r = sqrtf(max(0.f, 1.f - z * z));

    float phi = 2.f * GPUPIf * rng.nextFloat();

    float3 n = make_float3(r * cosf(phi), r * sinf(phi), z);

    float3 p = s.getCenter1() + s.getRadius() * n;

    float3 wi = normalize(p - p_point);

    float dist2 = length2(p - p_point);

    LightSample ls{};

    float cosThetaLight = max(dot(n, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float area = 4.f * GPUPIf * s.getRadius() * s.getRadius();

    float pdf_area = 1.f / area;

    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = sqrtf(dist2);
    ls.radiance = m.color() * m.intensity();

    ls.pdf = pdf;
    //ls.normal = n;

    return ls;
}

DEVICE 
LightSample Light::sampleMeshGeom(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    const TriangleMesh &mesh = scene.triangleMeshes[getGeomIndex()];

    const Material &m = scene.materials[mesh.materialIndex];

    LightSample ls{};

    if (mesh.triangleCount == 0 || mesh.meshArea <= 0.f)
    {
        return ls;
    }

    float sampleArea = rng.nextFloat() * mesh.meshArea;

    int triIndex = 0;

    while (triIndex < mesh.triangleCount - 1 && mesh.triangleAreaCdf[triIndex] < sampleArea)
    {
        ++triIndex;
    }

    const uint3 &tri = mesh.triangles[triIndex];

    const float3 *vertices = mesh.vertices;

    float3 v0 = vertices[tri.x];
    float3 v1 = vertices[tri.y];
    float3 v2 = vertices[tri.z];

    float u = rng.nextFloat();
    float v = rng.nextFloat();

    if (u + v > 1.f)
    {
        u = 1.f - u;
        v = 1.f - v;
    }

    float3 p = v0 + u * (v1 - v0) + v * (v2 - v0);

    float3 wi = normalize(p - p_point);

    float dist2 = length2(p - p_point);

    float3 n = normalize(cross(v1 - v0, v2 - v0));

    float cosThetaLight = max(dot(n, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float pdf_area = 1.f / mesh.meshArea;

    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = sqrtf(dist2);

    ls.radiance = m.color() * m.intensity();

    ls.pdf = pdf;
    //ls.normal = n;

    return ls;
}

DEVICE 
LightSample Light::sample(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    switch (getType())
    {
    case SPHERE_GEOM:
        return sampleSphereGeom(p_point, rng, scene);

    case IMPLICIT_SPHERE_GEOM:
        return sampleImplicitSphereGeom(p_point, rng, scene);

    case MESH_GEOM:
        return sampleMeshGeom(p_point, rng, scene);

    default:
        return sample(p_point, rng);
    }
}

DEVICE 
LightSample Light::sample(const float3 &p_point, RNG&rng) const
{
    switch (getType())
    {
    case QUAD:
        return sampleQuad(p_point, rng);

    case CYLINDER:
        return sampleCylinder(p_point, rng);

    case SUN:
        return sampleCone(p_point, rng);

    default:
        return sample(p_point);
    }
}

DEVICE 
LightSample Light::sample(const float3 &p_point) const
{
    switch (getType())
    {
    case DIRECTIONNAL:
        return sampleDirectionnal(p_point);

    default:
        return samplePoint(p_point);
    }
}