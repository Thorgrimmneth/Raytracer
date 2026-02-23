#include "cuda_whitted_integrator.cuh"

    

    __device__
    float3 WhittedIntegrator::lighting(
        const CudaScene& scene,
        const Ray& primaryRay,
        const float tMin,
        const float tMax,
        curandState* rng
    )
    {
        const int MAX_STACK = 32;

        Ray rayStack[MAX_STACK];
        float3 weightStack[MAX_STACK];
        bool insideStack[MAX_STACK];
        int depthStack[MAX_STACK];

        int stackPtr = 0;

        // push primary ray
        rayStack[stackPtr] = primaryRay;
        weightStack[stackPtr] = float3f(1.f);
        insideStack[stackPtr] = false;
        depthStack[stackPtr] = 0;
        stackPtr++;

        float3 finalColor = float3f(0.f);

        while (stackPtr > 0)
        {
            stackPtr--;

            Ray ray = rayStack[stackPtr];
            float3 throughput = weightStack[stackPtr];
            bool isInside = insideStack[stackPtr];
            int depth = depthStack[stackPtr];

            HitRecord hit;

            if (!scene.intersect(ray, tMin, tMax, hit))
            {
                if(hit.distance <0){
                    finalColor += make_float3(1.f, 0.f, 0.f);
                }
                else{
                //verified
                float3 sky = toneMap(getSkyColor(ray));
                finalColor += throughput * sky;
                //finalColor += make_float3(1.f, 0.f, 0.f);
                }
                continue;
            }
            if (hit.materialIndex < 0 || hit.materialIndex >= scene.nbMaterials)
            {
                continue;
            }
            const Material& mtl = scene.materials[hit.materialIndex];
            

            // ---- MIRROR ----
            if (mtl.type == MIRROR)
            {
                if (stackPtr >= MAX_STACK - 1 || depth >= nbBounces - 1) continue;

                float3 reflDir = reflect(ray.direction, hit.normal);
                Ray reflRay(hit.point, reflDir);
                reflRay.offset(hit.normal);

                rayStack[stackPtr] = reflRay;
                weightStack[stackPtr] = throughput;
                insideStack[stackPtr] = isInside;
                depthStack[stackPtr] = depth + 1;
                stackPtr++;
            }

            // ---- TRANSPARENT ----
            else if (mtl.type == TRANSPARENT)
            {
                if (depth >= nbBounces - 1)
                    continue;

                float n1 = 1.f;
                float n2 = mtl.ior;
                float3 normal = hit.normal;

                if (isInside)
                {
                    float tmp = n1;
                    n1 = n2;
                    n2 = tmp;
                }

                float3 rayDir = ray.direction;

                float cosI = dot(normal, -rayDir);
                cosI = clamp(cosI, -1.f, 1.f);

                float sinT = (n1 / n2) * sqrtf(max(0.f, 1.f - cosI * cosI));

                // ---- TOTAL INTERNAL REFLECTION ----
                if (sinT > 1.f)
                {
                    float3 reflectedDir = reflect(rayDir, normal);
                    Ray reflectedRay(hit.point, reflectedDir);
                    reflectedRay.offset(normal);

                    rayStack[stackPtr] = reflectedRay;
                    weightStack[stackPtr] = throughput;
                    insideStack[stackPtr] = isInside;
                    depthStack[stackPtr] = depth + 1;
                    stackPtr++;
                    continue;
                }

                // Clamp for safety
                sinT = clamp(sinT, -1.f, 1.f);
                float cosT = sqrtf(max(0.f, 1.f - sinT * sinT));
                cosT = clamp(cosT, -1.f, 1.f);

                // Fresnel exact (like CPU)
                float rs = ((n1 * cosI) - (n2 * cosT)) /
                        ((n1 * cosI) + (n2 * cosT));
                rs = rs * rs;

                float rp = ((n1 * cosT) - (n2 * cosI)) /
                        ((n1 * cosT) + (n2 * cosI));
                rp = rp * rp;

                float reff = (rs + rp) * 0.5f;

                // ---- REFLECTION ----
                float3 reflectedDir = reflect(rayDir, normal);
                Ray reflectedRay(hit.point, reflectedDir);
                reflectedRay.offset(normal);

                // ---- REFRACTION ----
                float3 refractedDir = refract(rayDir, normal, n1 / n2);
                Ray refractedRay(hit.point, refractedDir);
                refractedRay.offset(-normal);

                // Push reflection
                rayStack[stackPtr] = reflectedRay;
                weightStack[stackPtr] = throughput * reff;
                insideStack[stackPtr] = isInside;
                depthStack[stackPtr] = depth + 1;
                stackPtr++;

                // Push refraction
                rayStack[stackPtr] = refractedRay;
                weightStack[stackPtr] = throughput * (1.f - reff);
                insideStack[stackPtr] = !isInside;
                depthStack[stackPtr] = depth + 1;
                stackPtr++;
            }

            // ---- DIFFUSE ----
            else
            {
                DirectLightingIntegrator integrator;
                float3 direct = integrator.directLighting(
                    scene, ray, hit, tMin, tMax, rng
                );

                finalColor += throughput * direct;
            }
        }

        return finalColor;
    }

    __device__
    float3 WhittedIntegrator::toneMap( const float3 & c ){
		return (c * exposure) / ( float3f( 1.f ) + c );
	}

    __device__
    float3 WhittedIntegrator::getSkyColor( const Ray & p_ray )
	{
		float3 rayDir = normalize( p_ray.direction );
		float3 sunDir = normalize( sunDirection );

		float tMax = 60000.f;
		float dt   = tMax / skyColorSamples;

		float3 sumR = float3f(0.f);
		float3 sumM = float3f(0.f);

		float opticalDepthR = 0.f;
		float opticalDepthM = 0.f;

		float mu = dot( rayDir, sunDir );

		float phaseR = ( 3.f / ( 16.f * GPUPIf ) ) * ( 1.f + mu * mu );

		float g		 = 0.76f;
		float phaseM = ( 3.f / ( 8.f * GPUPIf ) ) * ( ( 1.f - g * g ) * ( 1.f + mu * mu ) )
					   / ( ( 2.f + g * g ) * pow( 1.f + g * g - 2.f * g * mu, 1.5f ) );

		float sunBelow = max( 0.f, -sunDir.y );

		for ( int i = 0; i < skyColorSamples; ++i )
		{
			float t = ( i + 0.5f ) * dt;
			float3 p = p_ray.pointAtT( t );

			float3 planetCenter = make_float3(0.f, -earthRadius, 0.f);
            float height = length(p - planetCenter) - earthRadius;
            height = max(0.f, height);

			float altitudeFade = exp( -height / 5.f );
			float horizonFade  = exp( -sunBelow * 20.f * altitudeFade );

			float nhr = exp( -height / hr ) * dt * horizonFade;
			float nhm = exp( -height / hm ) * dt * horizonFade;

			opticalDepthR += nhr;
			opticalDepthM += nhm;

			float sunOpticalDepthR = 0.f;
			float sunOpticalDepthM = 0.f;

			int	  sunSamples = 8;
			float sunDt		 = 100.f / sunSamples;

			for ( int j = 0; j < sunSamples; ++j )
			{
				float ts = ( j + 0.5f ) * sunDt;
				float3 ps = p + sunDir * ts;

				float h = length(ps - planetCenter) - earthRadius;
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
