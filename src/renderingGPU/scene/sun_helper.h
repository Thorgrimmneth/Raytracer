#pragma once

#include "../lights/light.cuh"
#include "scene_helper.cuh"

inline Light createSun(float3 sunDir, SceneHelper sceneHelper)
{
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    return l;
}