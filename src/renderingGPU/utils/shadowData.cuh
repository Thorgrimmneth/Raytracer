#pragma once

struct ShadowData {
    float3 *f;
    float *pdfBSDF;
    float *pdfLight;
    float3 *origin;
    float3 *direction;
    float *cosTheta;
    float *pixelIndex;
};