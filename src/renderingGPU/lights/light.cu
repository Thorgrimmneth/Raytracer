#include "../scene/scene.cuh"
#include "light.cuh"

DEVICE 
LightSample Light::sampleCylinder(const float3 &p_point, RNG&rng) const
{
    float u = rng.nextFloat();
    float v = rng.nextFloat();

    float3 dir = normalize(getDirection());

    // Use cross with (1,0,0) first, check length inline
    float3 uVec = cross(dir, make_float3(1.f, 0.f, 0.f));
    float uLen = length(uVec);
    if (uLen < 1e-3f)
    {
        uVec = cross(dir, make_float3(0.f, 1.f, 0.f));
        uVec = uVec * (1.f / length(uVec));
    }
    else
    {
        uVec = uVec * (1.f / uLen);
    }

    float3 vVec = cross(dir, uVec);  // Already normalized since dir and uVec are orthonormal

    float z = u * getHeight();
    float cosPhi = cosf(v * 2.f * GPUPIf);
    float sinPhi = sinf(v * 2.f * GPUPIf);

    float radius = getRadius();
    float3 pointOnCircle = radius * (cosPhi * uVec + sinPhi * vVec);

    float3 pos = getPosition();
    float3 randomPos = pos + z * dir + pointOnCircle;

    float3 normal = normalize(pointOnCircle);  // Normalize directly from pointOnCircle (no subtraction needed)

    float3 diff = randomPos - p_point;
    float dist2 = length2(diff);
    float3 lightDir = diff * (1.f / sqrtf(dist2));  // Compute sqrt once for both norm and distance

    float cosTheta = dot(normal, -lightDir);

    if (cosTheta <= 0.f)
    {
        LightSample s{};
        s.pdf = 0.f;
        return s;
    }

    float area = 2.f * GPUPIf * radius * (radius + getHeight());
    float pdf = dist2 / (area * cosTheta);

    LightSample sample{};
    sample.radiance = getColorPower();
    sample.pdf = pdf;
    sample.power = getIntensity();
    sample.distance = sqrtf(dist2);
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
    float3 diff = getPosition() - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 radiance = getColorPower() * (1.f / dist2);  // Use dist2 directly instead of dist*dist

    LightSample rep{};

    rep.radiance = radiance;
    rep.pdf = 1.f;
    rep.power = getIntensity();
    rep.distance = dist;
    rep.direction = diff * (1.f / dist);  // Normalize using pre-computed sqrt

    return rep;
}

DEVICE 
LightSample Light::sampleQuad(const float3 &p_point, RNG&rng) const
{
    float3 randomPos = getPosition() + rng.nextFloat() * getDirection() + rng.nextFloat() * getV();

    float3 diff = randomPos - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 lightDir = diff * (1.f / dist);  // Normalize using pre-computed sqrt

    float cosTheta = dot(getNormal(), -lightDir);

    if (cosTheta <= 0.f)
    {
        LightSample s{};
        s.pdf = 0.f;
        return s;
    }

    float3 crossVec = cross(getDirection(), getV());
    float area = length(crossVec);  // Compute area from cross product magnitude

    LightSample rep{};

    rep.radiance = getColorPower();
    rep.pdf = dist2 / (area * cosTheta);  // Use dist2 to avoid redundant multiplication

    rep.power = getIntensity();
    rep.distance = dist;
    rep.direction = lightDir;

    return rep;
}

DEVICE 
LightSample Light::sampleCone(const float3 &p_point, RNG&rng) const
{
    float sunAngularRadius = 3.f * GPUPIf / 180.f;
    float cosMax = cosf(sunAngularRadius);

    float u1 = rng.nextFloat();
    float u2 = rng.nextFloat();

    float cosTheta = 1.0f - u1 * (1.0f - cosMax);
    float sinTheta = sqrtf(max(0.f, 1.0f - cosTheta * cosTheta));  // Add max() for numerical safety

    float phi = 2.0f * GPUPIf * u2;
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    float3 w = normalize(getDirection());

    // Compute up vector - use ternary for better instruction pipelining
    float3 up = fabs(w.y) < 0.99f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

    float3 u = cross(up, w);
    float uLen = length(u);
    u = u * (1.f / uLen);  // Normalize

    float3 v = cross(w, u);  // Already normalized

    float3 sampledDir = u * (cosPhi * sinTheta) + v * (sinPhi * sinTheta) + w * cosTheta;
    // sampledDir is already normalized due to sin²θ + cos²θ = 1

    LightSample rep{};

    rep.direction = sampledDir;  // Safe normalization just in case
    rep.distance = 1e20f;
    rep.radiance = getColorPower();

    rep.pdf = 1.0f / (2.0f * GPUPIf * (1.0f - cosMax));

    rep.power = getIntensity();

    return rep;
}

DEVICE 
LightSample Light::sampleSphereGeom(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    const Sphere &s = scene.spheres[getMeshInstanceIndex()];
    const Material &m = scene.materials[s.materialIndex];

    float z = 1.f - 2.f * rng.nextFloat();
    float r = sqrtf(max(0.f, 1.f - z * z));  // Clamp to avoid NaN

    float phi = 2.f * GPUPIf * rng.nextFloat();
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    float3 n = make_float3(r * cosPhi, r * sinPhi, z);
    float radius = s.radius;
    float3 p = s.center1 + radius * n;

    float3 diff = p - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 wi = diff * (1.f / dist);  // Normalize using pre-computed sqrt

    LightSample ls{};

    float cosThetaLight = max(dot(n, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float area = 4.f * GPUPIf * radius * radius;
    float pdf_area = 1.f / area;
    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = dist;
    ls.radiance = m.color() * m.intensity();
    ls.pdf = pdf;

    return ls;
}

DEVICE 
LightSample Light::sampleImplicitSphereGeom(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    const ImplicitSphere &s = scene.implicitSpheres[getMeshInstanceIndex()];
    const Material &m = scene.materials[s.getMaterialIndex()];

    float z = 1.f - 2.f * rng.nextFloat();
    float r = sqrtf(max(0.f, 1.f - z * z));  // Clamp to avoid NaN

    float phi = 2.f * GPUPIf * rng.nextFloat();
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    float3 n = make_float3(r * cosPhi, r * sinPhi, z);
    float radius = s.getRadius();
    float3 p = s.getCenter1() + radius * n;

    float3 diff = p - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 wi = diff * (1.f / dist);  // Normalize using pre-computed sqrt

    LightSample ls{};

    float cosThetaLight = max(dot(n, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float area = 4.f * GPUPIf * radius * radius;
    float pdf_area = 1.f / area;
    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = dist;
    ls.radiance = m.color() * m.intensity();
    ls.pdf = pdf;

    return ls;
}

DEVICE 
LightSample Light::sampleMeshGeom(const float3 &p_point, RNG&rng, const CudaScene &scene) const
{
    const MeshInstance &inst = scene.meshInstances[getMeshInstanceIndex()];
    const MeshGeometry &geom = scene.meshGeometries[inst.geometryIndex];
    const Material &m = scene.materials[inst.materialIndex];

    LightSample ls{};

    if (geom.triangleCount == 0 || geom.meshArea <= 0.f)
    {
        return ls;
    }

    float sampleArea = rng.nextFloat() * geom.meshArea;

    int triIndex = 0;

    while (triIndex < geom.triangleCount - 1 && geom.triangleAreaCdf[triIndex] < sampleArea)
    {
        ++triIndex;
    }

    const uint3 &tri = geom.triangles[triIndex];
    const float3 *vertices = geom.vertices;

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

    float3 diff = p - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 wi = diff * (1.f / dist);  // Normalize using pre-computed sqrt

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 crossVec = cross(edge1, edge2);
    float3 n = crossVec * (1.f / length(crossVec));  // Normalize cross product

    float cosThetaLight = max(dot(n, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float pdf_area = 1.f / geom.meshArea;
    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = dist;
    ls.radiance = m.color() * m.intensity();
    ls.pdf = pdf;

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