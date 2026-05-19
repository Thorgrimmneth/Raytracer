#pragma once

struct cudaGraphicsResource;

class Renderer
{
public:

    Renderer();

    ~Renderer();

    void render(bool outputImage=true);
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

    void renderFrame(bool outputImage);

    unsigned char* getFramebuffer();

    void resetAccumulation();

    void cleanUp();

    void renderFrameWavefront(bool outputImage);

    void setInteropResource(cudaGraphicsResource* resource);

private:

    class Impl;

    Impl* impl;
};