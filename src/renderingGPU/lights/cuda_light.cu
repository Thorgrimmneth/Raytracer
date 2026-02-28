#include "cuda_light.cuh"
#include <curand_kernel.h>

__device__
LightSample Light::sampleCylinder(const float3 &p_point, curandState *rng) const
{
	float u = curand_uniform(rng);
	float v = curand_uniform(rng);

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
LightSample Light::sampleDirectionnal(const float3 &p_point) const
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
LightSample Light::samplePoint(const float3 &p_point) const
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
LightSample Light::sampleQuad(const float3 &p_point, curandState *rng) const
{
	float3 randomPos = position + curand_uniform(rng) * u + curand_uniform(rng) * v;
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
LightSample Light::sampleCone(const float3& p_point, curandState *rng) const
{
	float sunAngularRadius = 3.f * GPUPIf / 180.f;

    float u1 = curand_uniform(rng);
    float u2 = curand_uniform(rng);

    float cosTheta = 1.0f - u1 * (1.0f - cosf(sunAngularRadius));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float phi = 2.0f * GPUPIf * u2;

    float3 w = normalize(direction);
    float3 up = fabs(w.y) < 0.99f ? make_float3(0,1,0) : make_float3(1,0,0);
    float3 u = normalize(cross(up, w));
    float3 v = cross(w, u);

    float3 sampledDir =
        normalize(u * cosf(phi) * sinTheta +
                  v * sinf(phi) * sinTheta +
                  w * cosTheta);
	float cosMax = cosf(sunAngularRadius);
    LightSample rep;
    rep.direction = sampledDir;
    rep.distance  = 1e20f;
    rep.radiance  = color * power;
    //rep.pdf       = 1.0f / (2.0f * GPUPIf * (1.0f - cosMax));
	rep.pdf = 1.f;
    rep.power     = power;

    return rep;
}

__device__
LightSample Light::sample(const float3 &p_point, curandState *rng) const
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
		return samplePoint(p_point);
	}
}

__device__
LightSample Light::sample(const float3 &p_point) const
{
	switch (type)
	{
	case DIRECTIONNAL:
		return sampleDirectionnal(p_point);
	default:
		return samplePoint(p_point);
	}
}