#include "../scene/scene.cuh"
#include "light.cuh"

DEVICE LightSample Light::sampleCylinder(const float3 &p_point, RNG &rng) const
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

    float3 vVec = cross(dir, uVec); // Already normalized since dir and uVec are orthonormal

    float z = u * getHeight();
    float cosPhi = cosf(v * 2.f * GPUPIf);
    float sinPhi = sinf(v * 2.f * GPUPIf);

    float radius = getRadius();
    float3 pointOnCircle = radius * (cosPhi * uVec + sinPhi * vVec);

    float3 pos = getPosition();
    float3 randomPos = pos + z * dir + pointOnCircle;

    float3 normal = normalize(pointOnCircle); // Normalize directly from pointOnCircle (no subtraction needed)

    float3 diff = randomPos - p_point;
    float dist2 = length2(diff);
    float3 lightDir = diff * (1.f / sqrtf(dist2)); // Compute sqrt once for both norm and distance

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

DEVICE LightSample Light::sampleDirectionnal(const float3 &p_point) const
{
    LightSample rep{};

    rep.radiance = getColorPower();
    rep.pdf = 1.f;
    rep.power = getIntensity();
    rep.distance = 20000.f;
    rep.direction = normalize(-getDirection());

    return rep;
}

DEVICE LightSample Light::samplePoint(const float3 &p_point) const
{
    float3 diff = getPosition() - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 radiance = getColorPower() * (1.f / dist2); // Use dist2 directly instead of dist*dist

    LightSample rep{};

    rep.radiance = radiance;
    rep.pdf = 1.f;
    rep.power = getIntensity();
    rep.distance = dist;
    rep.direction = diff * (1.f / dist); // Normalize using pre-computed sqrt

    return rep;
}

DEVICE LightSample Light::sampleQuad(const float3 &p_point, RNG &rng) const
{
    // Sample on quad plane
    float u_rand = rng.nextFloat();
    float v_rand = rng.nextFloat();

    float3 dir = getDirection();
    float3 v_vec = getV();
    float3 randomPos = getPosition() + u_rand * dir + v_rand * v_vec;

    // Compute distance and direction to light
    float3 diff = randomPos - p_point;
    float dist2 = length2(diff);
    float dist = rsqrtf(dist2); // Use rsqrtf for reciprocal sqrt
    float3 lightDir = diff * dist;

    // Compute area via cross product magnitude
    float3 crossVec = cross(dir, v_vec);
    float area = length(crossVec);

    float3 normal = getNormal();
    float cosTheta = dot(normal, -lightDir);

    if (cosTheta <= 0.f)
    {
        LightSample s{};
        s.pdf = 0.f;
        return s;
    }

    // PDF: dist2 / (area * cosTheta)
    LightSample rep{};
    rep.radiance = getColorPower();
    rep.pdf = dist2 / (area * cosTheta);
    rep.power = getIntensity();
    rep.distance = 1.f / dist; // Compute actual distance from rsqrtf result
    rep.direction = lightDir;

    return rep;
}

DEVICE LightSample Light::sampleCone(const float3 &p_point, RNG &rng) const
{
    float sunAngularRadius = 3.f * GPUPIf / 180.f;
    float cosMax = cosf(sunAngularRadius);
    float invCosNorm = 1.0f / (2.0f * GPUPIf * (1.0f - cosMax));

    float u1 = rng.nextFloat();
    float u2 = rng.nextFloat();

    // Pre-compute trig for sampling angles
    float phi_u2 = 2.0f * GPUPIf * u2;
    float cosPhi = cosf(phi_u2);
    float sinPhi = sinf(phi_u2);

    float cosTheta = 1.0f - u1 * (1.0f - cosMax);
    float sinTheta_sq = max(0.f, 1.0f - cosTheta * cosTheta);
    float sinTheta = sqrtf(sinTheta_sq);

    float3 w = getDirection(); // getDirection() should already be normalized
    float3 up = fabs(w.y) < 0.99f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);

    float3 u_cross = cross(up, w);
    float u_len_inv = rsqrtf(max(1e-6f, length2(u_cross)));
    float3 u = u_cross * u_len_inv;

    float3 v = cross(w, u);

    float3 sampledDir = make_float3(u.x * (cosPhi * sinTheta) + v.x * (sinPhi * sinTheta) + w.x * cosTheta,
                                    u.y * (cosPhi * sinTheta) + v.y * (sinPhi * sinTheta) + w.y * cosTheta,
                                    u.z * (cosPhi * sinTheta) + v.z * (sinPhi * sinTheta) + w.z * cosTheta);

    LightSample rep{};
    rep.direction = sampledDir;
    rep.distance = 20000.f;
    rep.radiance = getColorPower();
    rep.pdf = invCosNorm;
    rep.power = getIntensity();

    return rep;
}

DEVICE LightSample Light::sampleSDFGeom(const float3 &p_point, RNG &rng, const SDF *scene_sdfs,
                                        const Material *scene_materials) const
{
    const SDF &sdf = scene_sdfs[getSDFIndex()];
    const Material &m = scene_materials[sdf.getMaterialIndex()];

    LightSample ls{};

    float3 randomPos, normal;
    sdf.samplePoint(randomPos, normal, rng);

    float3 diff = randomPos - p_point;
    float dist2 = length2(diff);
    float dist = sqrtf(dist2);
    float3 wi = diff * (1.f / dist); // Normalize using pre-computed sqrt

    float cosThetaLight = max(dot(normal, -wi), 0.f);

    if (cosThetaLight <= 0.f)
        return ls;

    float pdf_area = 1.f / sdf.getArea();
    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = dist;
    ls.radiance = m.color() * m.intensity();
    ls.pdf = pdf;

    return ls;
}

DEVICE LightSample Light::sampleMeshGeom(const float3 &p_point, RNG &rng, const MeshInstance *scene_mesh_instances,
                                         const MeshGeometry *scene_mesh_geometries,
                                         const Material *scene_materials) const
{
    const MeshInstance &inst = scene_mesh_instances[getMeshInstanceIndex()];
    const MeshGeometry &geom = scene_mesh_geometries[inst.geometryIndex];
    const Material &m = scene_materials[inst.materialIndex];

    LightSample ls{};
    if (geom.triangleCount == 0 || inst.worldArea <= 0.f)
        return ls;

    // Sample a triangle proportional to its area
    float sampleArea = rng.nextFloat() * inst.worldArea;

    int lo = 0;
    int hi = geom.triangleCount - 1;

    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;

        if (geom.triangleAreaCdf[mid] < sampleArea)
            lo = mid + 1;
        else
            hi = mid;
    }

    const int triIndex = lo;

    const uint3 &tri = geom.triangles[triIndex];

    const float3 v0 = geom.vertices[tri.x];
    const float3 v1 = geom.vertices[tri.y];
    const float3 v2 = geom.vertices[tri.z];

    // Uniform barycentric sampling
    float u = rng.nextFloat();
    float v = rng.nextFloat();

    if (u + v > 1.f)
    {
        u = 1.f - u;
        v = 1.f - v;
    }

    // Object-space sample
    float3 pObj = v0 + u * (v1 - v0) + v * (v2 - v0);

    // World-space sample
    float3 p = transformPoint(inst.transform, pObj);

    float3 diff = p - p_point;
    float dist2 = length2(diff);

    if (dist2 < 1e-12f)
        return ls;

    float invDist = rsqrtf(dist2);
    float dist = dist2 * invDist;
    float3 wi = diff * invDist;

    // Transform edges to world space and compute world normal
    float3 e1 = transformVector(inst.transform, v1 - v0);
    float3 e2 = transformVector(inst.transform, v2 - v0);

    float3 n = normalize(cross(e1, e2));

    float cosThetaLight = dot(n, -wi);

    if (cosThetaLight <= 0.f)
        return ls;

    float pdf = dist2 / (inst.worldArea * cosThetaLight);

    ls.direction = wi;
    ls.distance = dist;
    ls.radiance = m.color() * m.intensity();
    ls.pdf = pdf;

    return ls;
}

DEVICE LightSample Light::sample(const float3 &p_point, RNG &rng, const Scene &scene) const
{
    switch (getType())
    {
    case SDF_GEOM:
        return sampleSDFGeom(p_point, rng, scene.sdfGeometries.sdfs, scene.materials);

    case MESH_GEOM:
        return sampleMeshGeom(p_point, rng, scene.meshInstances, scene.meshGeometries, scene.materials);

    default:
        return sample(p_point, rng);
    }
}

DEVICE LightSample Light::sample(const float3 &p_point, RNG &rng) const
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

DEVICE LightSample Light::sample(const float3 &p_point) const
{
    switch (getType())
    {
    case DIRECTIONNAL:
        return sampleDirectionnal(p_point);

    default:
        return samplePoint(p_point);
    }
}