#pragma once

#include "../utils/defines_cpu.hpp"
#include <memory>
#include "utils/shading_data.cuh"
struct cudaGraphicsResource;

class Renderer
{
  public:
    
    Renderer();

    ~Renderer();

    // Setter
    void set_interop_resource(cudaGraphicsResource *resource);

    void reset_accumulation();

    // Getter
    int get_frame_number();
    unsigned char *get_frame_buffer();
    float3 *get_finalized_image();

    // Initialisation
    void init(int width, int height, float sunDirx, float sunDiry, float sunDirz, int rngManip = 0);

    // Post-processing
    void apply_bloom();

    float render(bool outputImage = true, bool convergence = false);
    
    void finalize_image_no_render(bool convergence = false);

    // Cleaner
    void clean_up();

  private:
    class Impl;

    std::unique_ptr<Impl> impl;
};