#pragma once

#include "../utils/defines.hpp"

struct cudaGraphicsResource;

class Renderer
{
  public:
    Renderer();

    ~Renderer();

    // Setter
    void setInteropResource(cudaGraphicsResource *resource);

    // Setter (kinda)
    void changeMode();

    void resetAccumulation();

    // Getter
    int getFrameNumber();
    unsigned char *getFramebuffer();
    float3 *getFinalizedImage();

    // Initialisation
    void init(int width, int height, float sunDirx, float sunDiry, float sunDirz);

    // Post-processing
    void applyBloom();

    // Renderer : megakernel and wavefront
    float renderFrame(bool outputImage, bool convergence = false);
    float renderFrameWavefront(bool outputImage, bool convergence = false);
    
    // Main function
    float render(bool outputImage = true, bool convergence = false);

    // Cleaner
    void cleanUp();

  private:
    class Impl;

    Impl *impl;
};