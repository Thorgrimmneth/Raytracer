#pragma once

#include "../defines.hpp"

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

    // Initialisation
    void init(int width, int height, float sunDirx, float sunDiry, float sunDirz);

    // Post-processing
    void applyBloom();

    // Renderer : megakernel and wavefront
    void renderFrame(bool outputImage);
    void renderFrameWavefront(bool outputImage);
    
    // Main function
    void render(bool outputImage = true);

    // Cleaner
    void cleanUp();

  private:
    class Impl;

    Impl *impl;
};