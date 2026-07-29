#pragma once

#include "../scene/scene.cuh"
#include "../utils/op.cuh"


D_FORCEINLINE
int selectLightByImportance(const int &nbLights, const float *lightCumulativeWeights, RNG &rng)
{
    // Use pre-computed cumulative weights for O(log n) binary search
    if (nbLights <= 1)
        return 0;
    
    // Get total weight from last entry
    float totalWeight = lightCumulativeWeights[nbLights - 1];
    
    if (totalWeight <= 0.0f)
    {
        // Fallback to uniform selection if no lights have intensity
        return min(int(rng.nextFloat() * nbLights), nbLights - 1);
    }
    
    // Binary search in cumulative weights array
    float random = rng.nextFloat() * totalWeight;
    
    int left = 0;
    int right = nbLights - 1;
    
    while (left < right)
    {
        int mid = (left + right) / 2;
        if (lightCumulativeWeights[mid] < random)
            left = mid + 1;
        else
            right = mid;
    }
    
    return left;
}

// Helper function: Compute probability of selecting a specific light
D_FORCEINLINE
float getLightProbability(const int &nbLights, float *lightProbabilities, int lightIndex)
{
    // Use pre-computed light probabilities
    if (lightIndex < 0 || lightIndex >= nbLights)
        return 1.0f;
    
    if (nbLights <= 0)
        return 1.0f;
    
    // Return pre-computed probability
    return lightProbabilities[lightIndex];
}