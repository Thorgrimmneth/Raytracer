#include "../scene.cuh"
#include "light.cuh"


__device__
	LightSample
	Light::sampleCylinder(const float3 &p_point, RNG *rng) const
{
	float u = rng->nextFloat();
	float v = rng->nextFloat();

	float3 uVec = normalize(cross(direction, make_float3(1.f, 0.f, 0.f)));
	if (length(uVec) < 1e-3f)
		uVec = normalize(cross(direction, make_float3(0.f, 1.f, 0.f)));

	float3 vVec = normalize(cross(direction, uVec));

	float z = u * height;
	float phi = v * 2.f * GPUPIf;

	float3 pointOnCircle = radius * (cosf(phi) * uVec + sinf(phi) * vVec);
	float3 randomPos = position + z * direction + pointOnCircle;

	float3 axisPoint = position + z * direction;
	float3 normal = normalize(randomPos - axisPoint);

	float3 direction = normalize(randomPos - p_point);
	float dist = distance(p_point, randomPos);

	float cosTheta = dot(normal, -direction);
	if (cosTheta <= 0.f)
	{
		LightSample s;
		s.pdf = 0.f;
		return s;
	}

	float pdf = (dist * dist) / (area * cosTheta);

	float3 radiance = (color * power);
	LightSample sample;
	sample.radiance = radiance;
	sample.pdf = pdf;
	sample.power = power;
	sample.distance = dist;
	sample.direction = direction;

	return sample;
}

__device__
	LightSample
	Light::sampleDirectionnal(const float3 &p_point) const
{

	LightSample rep;
	rep.radiance = color * power;
	rep.pdf = 1.f;
	rep.power = power;
	rep.distance = 1e20f;
	rep.direction = normalize(-direction);

	return rep;
}

__device__
	LightSample
	Light::samplePoint(const float3 &p_point) const
{
	float dist = distance(p_point, position);
	float3 radiance = color * power / (dist * dist);

	LightSample rep;
	rep.radiance = radiance;
	rep.pdf = 1.f;
	rep.power = power;
	rep.distance = dist;
	rep.direction = normalize(position - p_point);

	return rep;
}

__device__
	LightSample
	Light::sampleQuad(const float3 &p_point, RNG *rng) const
{
	float3 randomPos = position + rng->nextFloat() * u + rng->nextFloat() * v;
	float3 direction = normalize(randomPos - p_point);
	float dist = distance(p_point, randomPos);
	float cosTheta = dot(normal, -direction);
	if (cosTheta <= 0.f)
	{
		LightSample s;
		s.pdf = 0.f;
		return s;
	}
	float3 radiance = (color * power);

	LightSample rep;
	rep.radiance = radiance;
	rep.pdf = (dist * dist) / (area * cosTheta);
	rep.power = power;
	rep.distance = dist;
	rep.direction = direction;

	return rep;
}

__device__
	LightSample
	Light::sampleCone(const float3 &p_point, RNG *rng) const
{
	float sunAngularRadius = 3.f * GPUPIf / 180.f;

	float u1 = rng->nextFloat();
	float u2 = rng->nextFloat();

	float cosTheta = 1.0f - u1 * (1.0f - cosf(sunAngularRadius));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	float phi = 2.0f * GPUPIf * u2;

	float3 w = normalize(direction);
	float3 up = fabs(w.y) < 0.99f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
	float3 u = normalize(cross(up, w));
	float3 v = cross(w, u);

	float3 sampledDir =
		normalize(u * cosf(phi) * sinTheta +
				  v * sinf(phi) * sinTheta +
				  w * cosTheta);
	float cosMax = cosf(sunAngularRadius);
	LightSample rep;
	rep.direction = sampledDir;
	rep.distance = 1e20f;
	rep.radiance = color * power;
	rep.pdf       = 1.0f / (2.0f * GPUPIf * (1.0f - cosMax));
	//rep.pdf = 1.f;
	rep.power = power;

	return rep;
}

__device__
	LightSample
	Light::sampleSphereGeom(const float3 &p_point, RNG *rng, const CudaScene& scene) const
{
	const Sphere& s = scene.spheres[geomIndex];
    const Material& m = scene.materials[s.materialIndex];

    float z = 1.f - 2.f * rng->nextFloat();
    float r = sqrtf(max(0.f, 1.f - z*z));
    float phi = 2.f * M_PI * rng->nextFloat();

    float3 n = make_float3(r*cos(phi), r*sin(phi), z);

    float3 p = s.center1 + s.radius * n;

    float3 wi = normalize(p - p_point);
    float dist2 = length2(p - p_point);

	LightSample ls;
    float cosThetaLight = max(dot(n, -wi), 0.f);
    if (cosThetaLight <= 0.f)
        return ls;

    float area = 4.f * M_PI * s.radius * s.radius;
    float pdf_area = 1.f / area;

    float pdf = pdf_area * dist2 / cosThetaLight;

    ls.direction = wi;
    ls.distance = sqrtf(dist2);
    ls.radiance = m.color() * m.intensity();
    ls.pdf = pdf;
    ls.normal = n;

    return ls;
}

__device__
LightSample
Light::sampleMeshGeom(const float3 &p_point, RNG *rng, const CudaScene& scene) const
{
    const TriangleMesh& mesh = scene.triangleMeshes[geomIndex];
    const Material& m = scene.materials[mesh.materialIndex];

    LightSample ls;
    if (mesh.triangleCount == 0 || mesh.meshArea <= 0.f)
        return ls;

    float sampleArea = rng->nextFloat() * mesh.meshArea;
    int triIndex = 0;
    while (triIndex < mesh.triangleCount - 1 && mesh.triangleAreaCdf[triIndex] < sampleArea)
        ++triIndex;

    const TriangleMeshGeometry& tri = mesh.triangles[triIndex];
    const float3* vertices = mesh.vertices;

    float3 v0 = vertices[tri.i0];
    float3 v1 = vertices[tri.i1];
    float3 v2 = vertices[tri.i2];

    float u = rng->nextFloat();
    float v = rng->nextFloat();

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
    ls.normal = n;

    return ls;
}

__device__
LightSample
Light::sample(const float3& p_point, RNG *rng, const CudaScene& scene) const
{
	switch(type)
	{
	case SPHERE_GEOM:
		return sampleSphereGeom(p_point, rng, scene);
	case MESH_GEOM:
		return sampleMeshGeom(p_point, rng, scene);
	default:
		return sample(p_point,rng);
	}
}

__device__
	LightSample
	Light::sample(const float3 &p_point, RNG *rng) const
{
	switch (type)
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

__device__
	LightSample
	Light::sample(const float3 &p_point) const
{
	switch (type)
	{
	case DIRECTIONNAL:
		return sampleDirectionnal(p_point);
	default:
		return samplePoint(p_point);
	}
}