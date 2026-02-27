#include "cuda_sphere.cuh"

__device__
bool Sphere::intersectGeometry(const Ray &ray, float& t1, float& t2) const
{
    float3 c0 = center1;
    float3 c1 = center2;

    float3 current_center = c0 + ray.time * (c1 - c0);

    float3 oc = ray.origin - current_center;

    float a = dot(ray.direction, ray.direction);
    float b = 2.f * dot(ray.direction, oc);
    float c = dot(oc, oc) - radius * radius;

    float delta = b*b - 4.f*a*c;
    if (delta < 0.f)
        return false;

    float sqrtDelta = sqrtf(delta);
    t1 = (-b - sqrtDelta) / (2.f * a);
    t2 = (-b + sqrtDelta) / (2.f * a);

    return true;
}

__device__
bool Sphere::intersect(const Ray &p_ray, const float p_tMin, const float p_tMax, HitRecord &p_hitRecord) const
{
  float t1;
  float t2;
  if (intersectGeometry(p_ray, t1, t2))
  {
    if (t1 > p_tMax)
    {
      return false;
    } // first intersection too far
    if (t1 < p_tMin)
    {
      t1 = t2;
    } // first intersection too near, check second one
    if (t1 < p_tMin || t1 > p_tMax)
    {
      return false;
    } // not in range

    // Intersection found, fill p_hitRecord.
    p_hitRecord.point = p_ray.pointAtT(t1);
    p_hitRecord.normal = computeNormal(p_hitRecord.point, p_ray.time);
    p_hitRecord.faceNormal(p_ray.direction);
    p_hitRecord.distance = t1;
    p_hitRecord.materialIndex = materialIndex;
    
    return true;
  }
  return false;
}

__device__
bool Sphere::intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const
{
  float t1;
  float t2;
  if (intersectGeometry(p_ray, t1, t2))
  {
    if (t1 > p_tMax)
    {
      return false;
    } // first intersection too far
    if (t1 < p_tMin)
    {
      t1 = t2;
    } // first intersection too near, check second one
    if (t1 < p_tMin || t1 > p_tMax)
    {
      return false;
    } // not in range
    return true;
  }
  return false;
}

__device__
    float3
    Sphere::computeNormal(const float3 &point, const double time) const
{
  float3 c0 = center1;
  float3 c1 = center2;

  float3 center = c0 + time * (c1 - c0);

  return normalize(point - center);
}