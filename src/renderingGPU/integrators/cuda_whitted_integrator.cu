#include "cuda_whitted_integrator.cuh"



__device__ __noinline__ 
static Ray getTransparent(const Material &mtl, float3 point, float3 normal, bool &isInside, float3 direction, curandState *rng)
{
    float n1 = isInside ? mtl.ior : 1.f;
    float n2 = isInside ? 1.f : mtl.ior;

    float3 dir = direction;

    float cosI = clamp(dot(normal, -dir), -1.f, 1.f);
    float eta = n1 / n2;

    float k = 1.f - eta * eta * (1.f - cosI * cosI);

    // Total internal reflection
    if (k < 0.f)
    {
        Ray ray = Ray(point, reflect(dir, normal));
        ray.offset(normal);
        return ray;
    }

    float cosT = sqrtf(k);

    float rs = ((n1 * cosI) - (n2 * cosT)) /
               ((n1 * cosI) + (n2 * cosT));
    rs *= rs;

    float rp = ((n1 * cosT) - (n2 * cosI)) /
               ((n1 * cosT) + (n2 * cosI));
    rp *= rp;

    float reff = 0.5f * (rs + rp);

    float xi = curand_uniform(rng);

    if (xi < reff)
    {
        // reflect
        Ray ray = Ray(point, reflect(dir, normal));
        ray.offset(normal);
        return ray;
    }
    else
    {
        // refract
        float3 refrDir = eta * dir + (eta * cosI - cosT) * normal;
        Ray ray = Ray(point, refrDir);
        ray.offset(-normal);

        isInside = !isInside;
        return ray;
    }
}

__device__
PixelData WhittedIntegrator::lighting(
        const CudaScene &scene,
        const Ray &primaryRay,
        const float tMin,
        const float tMax,
        curandState *rng)
{
    PixelData pixelData;

    float3 finalColor = make_float3(0.f);

    Ray ray = primaryRay;
    float3 throughput = make_float3(1.f);
    bool isInside = false;

    for (int depth = 0; depth < nbBounces; depth++)
    {
        HitRecord hit;

        if (!scene.intersect(ray, tMin, tMax, hit))
        {
            
            finalColor += throughput * toneMap(getSkyColor(ray));
            if(depth == 0){
                pixelData.albedo = finalColor;
                pixelData.depth = tMax;
                pixelData.normal = -ray.direction;
            }
            break;
        }

        const Material &mtl = scene.materials[hit.materialIndex];

        // ---------------- MIRROR ----------------
        if (mtl.type == MIRROR)
        {
            ray = Ray(hit.point, reflect(ray.direction, hit.normal));
            ray.offset(hit.normal);
            if(depth == 0){
                pixelData.albedo = make_float3(0.f);
                pixelData.depth = hit.distance;
                pixelData.normal = hit.normal;
            }
            continue;
        }

        // ---------------- TRANSPARENT ----------------
        else if (mtl.type == TRANSPARENT)
        {
            ray = getTransparent(mtl, hit.point, hit.normal, isInside, ray.direction, rng);
            if(depth == 0){
                pixelData.albedo = make_float3(0.f);
                pixelData.depth = hit.distance;
                pixelData.normal = hit.normal;
            }
            continue;
        }
        // ---------------- DIFFUSE ----------------
        float3 direct =
            DirectLightingIntegrator::directLighting(
                scene, ray, hit, tMin, tMax, rng);
        if(depth == 0){
                pixelData.albedo = mtl.color;
                pixelData.depth = hit.distance;
                pixelData.normal = hit.normal;
            }
        finalColor += throughput * direct;
        break;
    }
    pixelData.radiance = finalColor;
    pixelData.normal = normalize(pixelData.normal);
    return pixelData;
}

__device__ __forceinline__
float3 WhittedIntegrator::toneMap(const float3 &c)
{
    float3 c1 = c * exposure;
    return (c1 * exposure) / (make_float3(1.f) + c1);
}

__device__ __noinline__
float3 WhittedIntegrator::getSkyColor(const Ray &ray)
{
    float3 rayDir = normalize(ray.direction);
    float3 sunDir = normalize(sunDirection);
    float mult = lerp(1.f, 20.f, (max(-0.4f,sunDir.y) + 0.4)/1.4f);

    float segmentLength = sizeAtmosphere / skyColorSamples;
    float tCurrent = 0.0f;

    float3 sumR = make_float3(0.f);
    float3 sumM = make_float3(0.f);

    float opticalDepthR = 0.0f;
    float opticalDepthM = 0.0f;

    float mu = dot(rayDir, sunDir);

    float g = 0.95f;

    float phaseR = (3.0f / (16.0f * GPUPIf)) * (1.0f + mu * mu);

    float temp = 1.0f + g * g - 2.0f * g * mu;
    float phaseM = (3.0f / (8.0f * GPUPIf)) *
                   ((1.0f - g * g) * (1.0f + mu * mu)) /
                   ((2.0f + g * g) * temp * sqrtf(temp));

    for (int i = 0; i < skyColorSamples; ++i)
    {
        float3 samplePosition = ray.origin + rayDir * (tCurrent + segmentLength * 0.5f);

        float height = max(samplePosition.y, 0.0f);

        float hrLocal = expf(-height / hr);
        float hmLocal = expf(-height / hm);

        opticalDepthR += hrLocal * segmentLength;
        opticalDepthM += hmLocal * segmentLength;

        float3 sunSamplePosition = samplePosition;

        float opticalDepthLightR = 0.0f;
        float opticalDepthLightM = 0.0f;

        const int sunSamples = 8;
        float sunSegmentLength = sizeAtmosphere / sunSamples;

        for (int j = 0; j < sunSamples; ++j)
        {
            sunSamplePosition += sunDir * sunSegmentLength;

            float heightLight = max(sunSamplePosition.y, 0.0f);

            opticalDepthLightR += expf(-heightLight / hr) * sunSegmentLength;
            opticalDepthLightM += expf(-heightLight / hm) * sunSegmentLength;
        }

        float3 tau =
            betaR * (opticalDepthR + opticalDepthLightR) +
            betaM * (opticalDepthM + opticalDepthLightM);

        float3 attenuation =
            make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));

        sumR += attenuation * hrLocal * segmentLength;
        sumM += attenuation * hmLocal * segmentLength;

        tCurrent += segmentLength;
    }

    float3 sky = sumR * betaR * phaseR +
                 sumM * betaM * phaseM * 0.3f;

    float sunAngularRadius = 0.00465f;
    float cosTheta = dot(rayDir, sunDir);

    float sunDisk =
        smoothstep(cos(sunAngularRadius),
                   cos(sunAngularRadius * 0.5f),
                   cosTheta);

    float3 sunColor = make_float3(30.f, 27.f, 24.f);

    sky += sunColor * sunDisk;

    return sky * mult;
}
