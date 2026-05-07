#pragma once

struct cudaGraphicsResource;

class Renderer
{
public:

    Renderer();

    ~Renderer();

    void init(
        int width,
        int height,
        float sunDirx,
        float sunDiry,
        float sunDirz
    );

    void applyBloom();

    int getFrameNumber();

    void renderFrame();

    unsigned char* getFramebuffer();

    void resetAccumulation();

    void cleanup();

    void setInteropResource(cudaGraphicsResource* resource);

private:

    class Impl;

    Impl* impl;
};