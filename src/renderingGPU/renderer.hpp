#pragma once

struct cudaGraphicsResource;

class Renderer
{
public:

    Renderer();

    ~Renderer();

    void render();
    void init(
        int width,
        int height,
        float sunDirx,
        float sunDiry,
        float sunDirz
    );

    void changeMode();
    
    void applyBloom();

    int getFrameNumber();

    void renderFrame();

    unsigned char* getFramebuffer();

    void resetAccumulation();

    void cleanup();

    void renderFrameWavefront();

    void setInteropResource(cudaGraphicsResource* resource);

private:

    class Impl;

    Impl* impl;
};