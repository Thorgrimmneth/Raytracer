#pragma once

#define GPUPIf 3.141592654f
#define GPUInvPIf (1.f/GPUPIf)

extern __constant__ int nbBounces;
extern __constant__ float earthRadius;
extern __constant__ float3 sunDirection;
extern __constant__ int skyColorSamples;
extern __constant__ float hr;
extern __constant__ float hm;
extern __constant__ float3 betaR;
extern __constant__ float3 betaM;
extern __constant__ float exposure;
extern __constant__ float sizeAtmosphere;