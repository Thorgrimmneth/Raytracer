#pragma once

struct Camera{
    float fov;
    float aspect;
    float focalDistance;
    float4 cameraPos;
    float4 topLeft;
    float4 viewPortU;
    float4 viewPortV;
};