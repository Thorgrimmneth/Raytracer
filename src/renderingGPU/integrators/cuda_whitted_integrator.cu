#include "cuda_whitted_integrator.cuh"

__device__
float3 WhittedIntegrator::lighting(
    const CudaScene& scene,
    const Ray& primaryRay,
    const float tMin,
    const float tMax,
    curandState* rng)
{
    float3 finalColor = float3f(0.f);

    Ray ray = primaryRay;
    float3 throughput = float3f(1.f);
    bool isInside = false;

    for (int depth = 0; depth < nbBounces; depth++)
    {
        HitRecord hit;

        if (!scene.intersect(ray, tMin, tMax, hit))
        {
            finalColor += throughput * toneMap(getSkyColor(ray));
            break;
        }

        const Material& mtl = scene.materials[hit.materialIndex];

        // ---------------- MIRROR ----------------
        if (mtl.type == MIRROR)
        {
            ray = Ray(hit.point, reflect(ray.direction, hit.normal));
            ray.offset(hit.normal);
            continue;
        }

        // ---------------- TRANSPARENT ----------------
        else if (mtl.type == TRANSPARENT)
        {
            float n1 = isInside ? mtl.ior : 1.f;
            float n2 = isInside ? 1.f : mtl.ior;

            float3 normal = hit.normal;
            float3 dir = ray.direction;

            float cosI = clamp(dot(normal, -dir), -1.f, 1.f);
            float eta = n1 / n2;

            float k = 1.f - eta * eta * (1.f - cosI * cosI);

            // Total internal reflection
            if (k < 0.f)
            {
                ray = Ray(hit.point, reflect(dir, normal));
                ray.offset(normal);
                continue;
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
                ray = Ray(hit.point, reflect(dir, normal));
                ray.offset(normal);

            }
            else
            {
                // refract
                float3 refrDir = eta * dir + (eta * cosI - cosT) * normal;
                ray = Ray(hit.point, refrDir);
                ray.offset(-normal);

                isInside = !isInside;
            }

            continue;
        }

        // ---------------- DIFFUSE ----------------
        float3 direct =
            DirectLightingIntegrator::directLighting(
                scene, ray, hit, tMin, tMax, rng);

        finalColor += throughput * direct;
        break;
    }

    return finalColor;
}

    __device__ __forceinline__
    float3 WhittedIntegrator::toneMap( const float3 & c ){
		return (c * exposure) / ( float3f( 1.f ) + c );
	}

    __device__ __noinline__
    float3 WhittedIntegrator::getSkyColor( const Ray & p_ray )
	{
        const float invHr = 1.f / hr;
        const float invHm = 1.f / hm;
		float3 rayDir = normalize( p_ray.direction );
		float3 sunDir = normalize( sunDirection );

		float dt   = sizeAtmosphere / skyColorSamples;

		float3 sumR = float3f(0.f);
		float3 sumM = float3f(0.f);

		float opticalDepthR = 0.f;
		float opticalDepthM = 0.f;

		float mu = dot( rayDir, sunDir );
		//float phaseR = ( 3.f / ( 16.f * GPUPIf ) ) * ( 1.f + mu * mu );
        float phaseR = 0.05968310365f * (1.f + mu * mu);
		/*float g		 = 0.76f;
        float temp = 1.f + g * g - 2.f * g * mu;
		float phaseM = ( 3.f / ( 8.f * GPUPIf ) ) * ( ( 1.f - g * g ) * ( 1.f + mu * mu ) )
					   / ( ( 2.f + g * g ) * temp * sqrt(temp) );*/
        float temp = 1.5776f - 1.56f * mu;
        float phaseM =  0.01956094267f * (1 + mu * mu) / (temp * sqrt(temp));

		for ( int i = 0; i < skyColorSamples; ++i )
		{
			float t = ( i + 0.5f ) * dt;
			float3 p = p_ray.pointAtT( t );

            float height = p.y;
            height = max(0.f, height);

			float nhr = exp( -height * invHr ) * dt;
			float nhm = exp( -height * invHm ) * dt;

			opticalDepthR += nhr;
			opticalDepthM += nhm;

			float sunOpticalDepthR = 0.f;
			float sunOpticalDepthM = 0.f;

			int	  sunSamples = 2;
			float sunDt		 = 100.f / sunSamples;
            
			for ( int j = 0; j < sunSamples; ++j )
			{
				float ts = ( j + 0.5f ) * sunDt;
				float3 ps = p + sunDir * ts;

				float h = ps.y;
                h = max(0.f, h);

				sunOpticalDepthR += exp( -h / hr ) * sunDt;
				sunOpticalDepthM += exp( -h / hm ) * sunDt;
			}

			float3 tau = betaR * ( opticalDepthR + sunOpticalDepthR ) + betaM * ( opticalDepthM + sunOpticalDepthM );

			float3 attenuation = make_float3( expf( -tau.x ), expf( -tau.y ), expf( -tau.z ) );

			sumR += attenuation * nhr;
			sumM += attenuation * nhm;
		}

		return ( sumR * betaR * phaseR + sumM * betaM * phaseM );
	}
