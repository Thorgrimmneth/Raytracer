#include "renderer.hpp"

#include "../devicePrograms/launch_radiance_params.cuh"
#include "camera/camera.cuh"
#include "renderingUtils/post_treatment.cuh"
#include "scene/scene.cuh"
#include "utils/constant.cuh"
#include "utils/fill_buffers.cuh"
#include "utils/optix_pass_data.cuh"
#include "utils/simplified_def.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <map>

__constant__ float EXPOSURE;
__constant__ Camera camera;

class Renderer::Impl
{
  public:
    int WIDTH = 1920;
    int HEIGHT = 1080;
    int MIN_SHADOW_RAY_NUMBER = WIDTH * HEIGHT;
    Camera H_CAMERA;

    const float THRESHOLD = 1.0f;
    const float BLOOM_STRENGTH = 0.25f;
    const float EXPOSURE = 1.0f;

    int sample_count = 0;
    int MAX_BOUNCES = 5;

    size_t BUFFER_SIZE_FLOAT3 = 0;

    float value = 1.f;
    float *d_value = nullptr;

    Scene scene;

    uint depth = 0;

    float3 sun_direction = make_float3(0.f); // Store sun direction for render passes

    OptixPassData<LaunchRadianceParams> radiance_pass;
    float3 *d_normalized_buffer = nullptr;
    float3 *d_hdrBloom_buffer = nullptr;
    float3 *d_bloom_buffer = nullptr;
    float3 *d_temp_buffer = nullptr;
    float3 *d_convergence_buffer = nullptr;
    float3 *d_accum_buffer = nullptr;

    int *d_isInside = nullptr;
    int *d_lastBounceWasDelta = nullptr; // Track if previous bounce was delta
    float *d_lastBsdfPdf = nullptr;

    float3 *d_origins = nullptr;
    float3 *d_directions = nullptr;
    float3 *d_throughputs = nullptr;
    RNG *d_rng = nullptr;

    dim3 block_size = dim3(16, 16);
    dim3 grid_size;

    cudaStream_t stream = nullptr;

    int w1 = 0;
    int h1 = 0;

    int w2 = 0;
    int h2 = 0;

    float3 *d_lvl1 = nullptr;
    float3 *d_lvl2 = nullptr;

    cudaGraphicsResource *cuda_texture_resource = nullptr;

    // CUDA event timing pairs
    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    std::map<std::string, float> kernel_times;
};

void Renderer::recordKernelTime(const std::string &kernel_name)
{
    cudaEventSynchronize(impl->event_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
    impl->kernel_times[kernel_name] += milliseconds;
}

Renderer::Renderer() : impl(std::make_unique<Impl>()) {}

Renderer::~Renderer() { clean_up(); }

// Helper

void Renderer::set_interop_resource(cudaGraphicsResource *resource) { impl->cuda_texture_resource = resource; }

void Renderer::reset_accumulation()
{
    impl->sample_count = 0;

    cudaMemset(impl->d_accum_buffer, 0, impl->BUFFER_SIZE_FLOAT3);
}

int Renderer::get_frame_number() { return impl->sample_count; }

HOST Camera init_camera(int width, int height)
{
    // ===== Camera =====
    float3 cam_pos = make_float3(0.f, 3.f, 8.f);
    float3 cam_target = make_float3(0.f, 1.f, 0.f);
    float3 cam_up = make_float3(0.f, 1.f, 0.f);

    float fov = 60.f;
    float aspect = (float)width / (float)height;
    float focal_distance = 1.f;

    // === Base vectors EXACTEMENT comme CPU ===
    float3 w = normalize(cam_pos - cam_target);
    float3 u = normalize(cross(cam_up, w));
    float3 v = normalize(cross(w, u));

    // === Viewport ===
    float theta = fov * 3.14159265f / 180.f;
    float viewport_height = 2.f * tanf(theta * 0.5f) * focal_distance;
    float viewport_width = viewport_height * aspect;

    float3 viewport_u = u * viewport_width;
    float3 viewport_v = v * viewport_height;

    float3 top_left = cam_pos - w * focal_distance + viewport_v * 0.5f - viewport_u * 0.5f;
    Camera camera = Camera{cam_pos, top_left, viewport_u, viewport_v};
    return camera;
}

HOST void init_constant(int width, int height, Camera c_camera, float4 sunDir)
{
    float c_exposure = 1.f;
    cudaMemcpyToSymbol(EXPOSURE, &c_exposure, sizeof(float));
    cudaMemcpyToSymbol(camera, &c_camera, sizeof(Camera));
    
}

void Renderer::init(int p_width, int p_height, float sunDirx, float sunDiry, float sunDirz, int rngManip)
{
    float global_size = sizeof(Impl);
    impl->H_CAMERA = init_camera(p_width, p_height);
    global_size += sizeof(Camera);
    float3 sunDir = normalize(make_float3(sunDirx, sunDiry, sunDirz));
    impl->sun_direction = sunDir;
    init_constant(p_width, p_height, impl->H_CAMERA, make_float4(sunDirx, sunDiry, sunDirz, 0.f));
    impl->WIDTH = p_width;
    impl->HEIGHT = p_height;
    setSeed(43);
    impl->scene = spheres(sunDir, impl->radiance_pass, global_size, rngManip);
    // impl->scene = showcase(sunDir, impl->radiance_pass, global_size);
     //impl->scene = loadScene(sunDir, impl->radiance_pass, global_size, rngManip);

    impl->BUFFER_SIZE_FLOAT3 = impl->WIDTH * impl->HEIGHT * sizeof(float3);

    // =========================
    // GPU buffers
    // =========================
    cudaEventCreate(&impl->event_start);
    cudaEventCreate(&impl->event_stop);
    cudaMalloc(&impl->d_normalized_buffer, impl->BUFFER_SIZE_FLOAT3);

    cudaMalloc(&impl->d_bloom_buffer, impl->BUFFER_SIZE_FLOAT3);
    cudaMalloc(&impl->d_hdrBloom_buffer, impl->BUFFER_SIZE_FLOAT3);
    cudaMalloc(&impl->d_temp_buffer, impl->BUFFER_SIZE_FLOAT3);
    cudaMalloc(&impl->d_accum_buffer, impl->BUFFER_SIZE_FLOAT3);
    cudaMalloc(&impl->d_convergence_buffer, impl->BUFFER_SIZE_FLOAT3);
    global_size += impl->BUFFER_SIZE_FLOAT3 * 6;
    cudaMalloc(&impl->d_value, sizeof(float));

    size_t pixel_count = impl->WIDTH * impl->HEIGHT;

    cudaMalloc(&impl->d_isInside, pixel_count * sizeof(int));
    cudaMalloc(&impl->d_lastBounceWasDelta, pixel_count * sizeof(int));
    cudaMalloc(&impl->d_lastBsdfPdf, pixel_count * sizeof(float));
    cudaMalloc(&impl->d_origins, pixel_count * sizeof(float3));
    cudaMalloc(&impl->d_directions, pixel_count * sizeof(float3));
    cudaMalloc(&impl->d_throughputs, pixel_count * sizeof(float3));
    cudaMalloc(&impl->d_rng, pixel_count * sizeof(RNG));
    global_size += pixel_count * sizeof(int) * 2; // d_keys and d_values

    // =========================
    // CUDA Stream for async operations
    // =========================

    cudaStreamCreate(&impl->stream);

    // =========================

    impl->block_size = dim3(16, 16);

    impl->grid_size = dim3((impl->WIDTH + impl->block_size.x - 1) / impl->block_size.x,
                           (impl->HEIGHT + impl->block_size.y - 1) / impl->block_size.y);

    // =========================
    // Bloom mip chain
    // =========================

    impl->w1 = impl->WIDTH / 2;
    impl->h1 = impl->HEIGHT / 2;

    impl->w2 = impl->w1 / 2;
    impl->h2 = impl->h1 / 2;

    cudaMalloc(&impl->d_lvl1, impl->w1 * impl->h1 * sizeof(float3));
    global_size += impl->w1 * impl->h1 * sizeof(float3);

    cudaMalloc(&impl->d_lvl2, impl->w2 * impl->h2 * sizeof(float3));
    global_size += impl->w2 * impl->h2 * sizeof(float3);
    printf("Renderer initialized with width: %d, height: %d, global_size: %.2f MB\n", impl->WIDTH, impl->HEIGHT,
           global_size / (1024.0f * 1024.0f));
    impl->sample_count = 0;
}

GLOBAL
void compare_buffers(const float3 *d_current_buffer, const float3 *d_previous_buffer, int width, int height,
                     float *value)
{
    // Block-level reduction without atomic operations for better performance
    __shared__ float block_sum;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        block_sum = 0.0f;

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float local_sum = 0.0f;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        float3 current = d_current_buffer[idx];
        float3 previous = d_previous_buffer[idx];
        float3 diff = abs(current - previous);
        local_sum = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // Write warp result to shared memory
    if (threadIdx.x % warpSize == 0)
        atomicAdd(&block_sum, local_sum);

    __syncthreads();

    // One thread writes block result
    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(value, block_sum);
}

// WAVEFRONT INIT
GLOBAL
void generatePrimaryRaysKernel(float3 *directions, RNG *p_rng, int width, int height, int sample_count, float invWidth,
                               float invHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixel_index = y * width + x;

    uint seed = (pixel_index * 0x9E3779B9u) ^ (sample_count * 0x6C078965u);
    RNG rng;
    rng.state = seed;

    float sx = (x + rng.nextFloat()) * invWidth;
    float sy = (y + rng.nextFloat()) * invHeight;
    p_rng[pixel_index] = rng;
    float3 ray_target = camera.top_left + sx * camera.view_port_u - sy * camera.view_port_v;
    directions[pixel_index] = normalize(ray_target - camera.camera_pos);
}

void Renderer::apply_bloom()
{
    // =========================
    // Extracts bright pixels, also normalizes
    // =========================

    extractBright<<<impl->grid_size, impl->block_size>>>(impl->d_accum_buffer, impl->d_normalized_buffer,
                                                         impl->d_bloom_buffer, impl->WIDTH, impl->HEIGHT,
                                                         impl->THRESHOLD, 1.f / (float)impl->sample_count);

    // =========================
    // Blur bloom
    // =========================

    applyMultiScaleBloom(impl->d_bloom_buffer, impl->d_temp_buffer, impl->d_lvl1, impl->d_lvl2, impl->w1, impl->h1,
                         impl->w2, impl->h2, impl->WIDTH, impl->HEIGHT);
}

float Renderer::render_no_text(bool outputImage, bool convergence)
{
    int pixel_count = impl->WIDTH * impl->HEIGHT;

    dim3 block2D = impl->block_size;
    dim3 grid2D = impl->grid_size;
    std::string resultValue = "Render result\n";
    dim3 block1D(256);

    cudaError_t err;

    // -------------------------------------------------------------------------
    // 1. Génération des rayons primaires + activeQueue
    // -------------------------------------------------------------------------
    float invWidth = 1.f / (float)(impl->WIDTH - 1);
    float invHeight = 1.f / (float)(impl->HEIGHT - 1);
    generatePrimaryRaysKernel<<<grid2D, block2D>>>(impl->d_directions, impl->d_rng, impl->WIDTH, impl->HEIGHT,
                                                   impl->sample_count, invWidth, invHeight);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "generatePrimaryRaysKernel error;" << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    initFloat3Buffer<<<blocks, threads>>>(impl->d_origins, pixel_count, impl->H_CAMERA.camera_pos);

    initFloat3Buffer<<<blocks, threads>>>(impl->d_throughputs, pixel_count, make_float3(1.f));

    initIntBuffer<<<blocks, threads>>>(impl->d_isInside, pixel_count, 0);

    initIntBuffer<<<blocks, threads>>>(impl->d_lastBounceWasDelta, pixel_count, 0); // First ray, no delta bounce yet

    initFloatBuffer<<<blocks, threads>>>(impl->d_lastBsdfPdf, pixel_count, 1.f);

    int h_active_count = pixel_count;
    impl->radiance_pass.params.maxBounces = impl->MAX_BOUNCES;
    impl->radiance_pass.params.sunDirection = impl->sun_direction;
    impl->radiance_pass.params.HR = 1.f / 7994.f;
    impl->radiance_pass.params.HM = 1.f / 1200.f;
    impl->radiance_pass.params.betaR = make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f);
    impl->radiance_pass.params.betaM = make_float3(21e-6f);
    impl->radiance_pass.params.atmosphereSize = 60000.f;
    impl->radiance_pass.params.nbSkySamples = 4;
    impl->radiance_pass.params.sunAngularRadius = cosf(2.1f * GPUPIf / 180.f);
    impl->radiance_pass.params.sunHalfAngularRadius = cosf(2.1f * GPUPIf / 180.f * 0.5f);

    // -------------------------------------------------------------------------
    // 2. Boucle wavefront activeQueue
    // -------------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // 2.1 Intersection uniquement des rayons actifs
    // ---------------------------------------------------------------------
    impl->radiance_pass.params.origins = impl->d_origins;
    impl->radiance_pass.params.directions = impl->d_directions;
    impl->radiance_pass.params.active_count = h_active_count;
    impl->radiance_pass.params.accum_buffer = impl->d_accum_buffer;
    impl->radiance_pass.params.throughputs = impl->d_throughputs;
    impl->radiance_pass.params.lastBounceWasDelta = impl->d_lastBounceWasDelta;
    impl->radiance_pass.params.isInside = impl->d_isInside;
    impl->radiance_pass.params.rngs = impl->d_rng;
    impl->radiance_pass.params.lastBsdfPdf = impl->d_lastBsdfPdf;

    cudaMemcpy(reinterpret_cast<void *>(impl->radiance_pass.d_params), &impl->radiance_pass.params,
               sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cerr << "CUDA ERROR before OptiX: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cerr << "CUDA EXECUTION ERROR before OptiX: " << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }
    OPTIX_CHECK(optixLaunch(impl->radiance_pass.pipeline,
                            0, // stream
                            impl->radiance_pass.d_params, sizeof(LaunchRadianceParams), &impl->radiance_pass.sbt,
                            h_active_count, 1, 1));

    // -------------------------------------------------------------------------
    // 3. Accumulation dans d_accum_buffer
    // -------------------------------------------------------------------------

    dim3 gridAllPixels((pixel_count + block1D.x - 1) / block1D.x);

    // -------------------------------------------------------------------------
    // 4. Incrément du sample count
    // -------------------------------------------------------------------------

    impl->sample_count++;

    // -------------------------------------------------------------------------
    // 6. Bloom + normalize
    // -------------------------------------------------------------------------
    apply_bloom();
    // -------------------------------------------------------------------------
    // 7. Mapping OpenGL / CUDA
    // -------------------------------------------------------------------------
    if (!outputImage)
    {
        finalize_image_no_render(convergence);
        return impl->value;
    }
    cudaArray_t textureArray;

    cudaGraphicsMapResources(1, &impl->cuda_texture_resource, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsMapResources error;" << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }
    cudaGraphicsSubResourceGetMappedArray(&textureArray, impl->cuda_texture_resource, 0, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsSubResourceGetMappedArray error;" << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);

        return -1.f;
    }

    cudaResourceDesc resource_desc;
    memset(&resource_desc, 0, sizeof(resource_desc));

    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObject = 0;

    cudaCreateSurfaceObject(&surfaceObject, &resource_desc);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaCreateSurfaceObject error;" << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);

        return -1.f;
    }

    // -------------------------------------------------------------------------
    // 8. Finalisation image
    // -------------------------------------------------------------------------
    finalizeImageV2<<<impl->grid_size, impl->block_size>>>(impl->d_normalized_buffer, impl->d_bloom_buffer,
                                                           impl->d_hdrBloom_buffer, surfaceObject, impl->WIDTH,
                                                           impl->HEIGHT, impl->EXPOSURE, impl->BLOOM_STRENGTH);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "finalizeImage error;" << cudaGetErrorString(err) << std::endl;
        cudaDestroySurfaceObject(surfaceObject);
        cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);
        return -1.f;
    }

    cudaDestroySurfaceObject(surfaceObject);

    cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsUnmapResources error;" << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compare_buffers<<<impl->grid_size, impl->block_size>>>(impl->d_hdrBloom_buffer, impl->d_convergence_buffer,
                                                               impl->WIDTH, impl->HEIGHT, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergence_buffer, impl->d_hdrBloom_buffer, impl->BUFFER_SIZE_FLOAT3,
                   cudaMemcpyDeviceToDevice);
    }

    return impl->value;
}

float Renderer::render_with_text(bool outputImage, bool convergence, bool outputText, std::string *result_numbers)
{
    int pixel_count = impl->WIDTH * impl->HEIGHT;

    dim3 block2D = impl->block_size;
    dim3 grid2D = impl->grid_size;
    dim3 block1D(256);

    cudaError_t err;

    // -------------------------------------------------------------------------
    // 1. Génération des rayons primaires + activeQueue
    // -------------------------------------------------------------------------
    float invWidth = 1.f / (float)(impl->WIDTH - 1);
    float invHeight = 1.f / (float)(impl->HEIGHT - 1);
    cudaEventRecord(impl->event_start);
    generatePrimaryRaysKernel<<<grid2D, block2D>>>(impl->d_directions, impl->d_rng, impl->WIDTH, impl->HEIGHT,
                                                   impl->sample_count, invWidth, invHeight);
    cudaEventRecord(impl->event_stop);
    cudaEventSynchronize(impl->event_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
    *result_numbers += "generatePrimaryRaysKernel;" + std::to_string(milliseconds) + "\n";

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "generatePrimaryRaysKernel error;" << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    int threads = 256;
    int blocks = (pixel_count + threads - 1) / threads;
    cudaEventRecord(impl->event_start);
    initFloat3Buffer<<<blocks, threads>>>(impl->d_origins, pixel_count, impl->H_CAMERA.camera_pos);

    initFloat3Buffer<<<blocks, threads>>>(impl->d_throughputs, pixel_count, make_float3(1.f));

    initIntBuffer<<<blocks, threads>>>(impl->d_isInside, pixel_count, 0);

    initIntBuffer<<<blocks, threads>>>(impl->d_lastBounceWasDelta, pixel_count, 0); // First ray, no delta bounce yet

    initFloatBuffer<<<blocks, threads>>>(impl->d_lastBsdfPdf, pixel_count, 1.f);

    cudaEventRecord(impl->event_stop);
    cudaEventSynchronize(impl->event_stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
    *result_numbers += "init buffers;" + std::to_string(milliseconds) + "\n";
    int h_active_count = pixel_count;
    // -------------------------------------------------------------------------
    // 2. Boucle wavefront activeQueue
    // -------------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // 2.1 Intersection uniquement des rayons actifs
    // ---------------------------------------------------------------------
    impl->radiance_pass.params.maxBounces = impl->MAX_BOUNCES;
    impl->radiance_pass.params.origins = impl->d_origins;
    impl->radiance_pass.params.directions = impl->d_directions;
    // impl->radiance_pass.params.hit_buffers = impl->hit_buffers;
    impl->radiance_pass.params.active_count = h_active_count;
    impl->radiance_pass.params.throughputs = impl->d_throughputs;
    impl->radiance_pass.params.lastBounceWasDelta = impl->d_lastBounceWasDelta;
    impl->radiance_pass.params.accum_buffer = impl->d_accum_buffer;
    impl->radiance_pass.params.isInside = impl->d_isInside;

    cudaMemcpy(reinterpret_cast<void *>(impl->radiance_pass.d_params), &impl->radiance_pass.params,
               sizeof(LaunchRadianceParams), cudaMemcpyHostToDevice);
    cudaEventRecord(impl->event_start);
    OPTIX_CHECK(optixLaunch(impl->radiance_pass.pipeline,
                            0, // stream
                            impl->radiance_pass.d_params, sizeof(LaunchRadianceParams), &impl->radiance_pass.sbt,
                            h_active_count, 1, 1));
    cudaEventRecord(impl->event_stop);
    cudaEventSynchronize(impl->event_stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
    *result_numbers += "radiance rays launch;" + std::to_string(milliseconds) + "\n";

    // ---------------------------------------------------------------------
    // 2.3 Récupération du nombre de rayons actifs pour le prochain bounce
    // ---------------------------------------------------------------------

    // if shadow rays still need to be launched, launch them now
    // -------------------------------------------------------------------------
    // 3. Accumulation dans d_accum_buffer
    // -------------------------------------------------------------------------

    dim3 gridAllPixels((pixel_count + block1D.x - 1) / block1D.x);

    // -------------------------------------------------------------------------
    // 4. Incrément du sample count
    // -------------------------------------------------------------------------

    impl->sample_count++;

    // -------------------------------------------------------------------------
    // 6. Bloom + normalize
    // -------------------------------------------------------------------------
    cudaEventRecord(impl->event_start);
    apply_bloom();
    cudaEventRecord(impl->event_stop);
    cudaEventSynchronize(impl->event_stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
    *result_numbers += "bloom;" + std::to_string(milliseconds) + "\n";
    // -------------------------------------------------------------------------
    // 7. Mapping OpenGL / CUDA
    // -------------------------------------------------------------------------
    if (!outputImage)
    {
        finalize_image_no_render(convergence);
        return impl->value;
    }
    cudaArray_t textureArray;

    cudaGraphicsMapResources(1, &impl->cuda_texture_resource, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsMapResources error;" << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }
    cudaGraphicsSubResourceGetMappedArray(&textureArray, impl->cuda_texture_resource, 0, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsSubResourceGetMappedArray error;" << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);

        return -1.f;
    }

    cudaResourceDesc resource_desc;
    memset(&resource_desc, 0, sizeof(resource_desc));

    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObject = 0;

    cudaCreateSurfaceObject(&surfaceObject, &resource_desc);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaCreateSurfaceObject error;" << cudaGetErrorString(err) << std::endl;

        cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);

        return -1.f;
    }

    // -------------------------------------------------------------------------
    // 8. Finalisation image
    // -------------------------------------------------------------------------
    cudaEventRecord(impl->event_start);
    finalizeImageV2<<<impl->grid_size, impl->block_size>>>(impl->d_normalized_buffer, impl->d_bloom_buffer,
                                                           impl->d_hdrBloom_buffer, surfaceObject, impl->WIDTH,
                                                           impl->HEIGHT, impl->EXPOSURE, impl->BLOOM_STRENGTH);
    cudaEventRecord(impl->event_stop);
    cudaEventSynchronize(impl->event_stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
    *result_numbers += "finalize;" + std::to_string(milliseconds) + "\n";
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "finalizeImage error;" << cudaGetErrorString(err) << std::endl;
        cudaDestroySurfaceObject(surfaceObject);
        cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);
        return -1.f;
    }

    cudaDestroySurfaceObject(surfaceObject);

    cudaGraphicsUnmapResources(1, &impl->cuda_texture_resource, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsUnmapResources error;" << cudaGetErrorString(err) << std::endl;
        return -1.f;
    }

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        cudaEventRecord(impl->event_start);
        compare_buffers<<<impl->grid_size, impl->block_size>>>(impl->d_hdrBloom_buffer, impl->d_convergence_buffer,
                                                               impl->WIDTH, impl->HEIGHT, impl->d_value);
        cudaEventRecord(impl->event_stop);
        cudaEventSynchronize(impl->event_stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, impl->event_start, impl->event_stop);
        *result_numbers += "convergence;" + std::to_string(milliseconds) + "\n";
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergence_buffer, impl->d_hdrBloom_buffer, impl->BUFFER_SIZE_FLOAT3,
                   cudaMemcpyDeviceToDevice);
    }

    return impl->value;
}
float Renderer::render(bool outputImage, bool convergence, bool outputText, std::string *result_numbers)
{
    if (outputText)
    {
        return render_with_text(outputImage, convergence, outputText, result_numbers);
    }
    else
    {
        return render_no_text(outputImage, convergence);
    }
}

void Renderer::finalize_image_no_render(bool convergence)
{
    // -------------------------------------------------------------------------
    // 8. Finalisation image
    // -------------------------------------------------------------------------

    finalizeImageV2NoRender<<<impl->grid_size, impl->block_size>>>(impl->d_normalized_buffer, impl->d_bloom_buffer,
                                                                   impl->d_hdrBloom_buffer, impl->WIDTH, impl->HEIGHT,
                                                                   impl->EXPOSURE, impl->BLOOM_STRENGTH);

    if (convergence)
    {
        impl->value = 0.f;
        cudaMemcpy(impl->d_value, &impl->value, sizeof(float), cudaMemcpyHostToDevice);

        // copy image for convergence
        compare_buffers<<<impl->grid_size, impl->block_size>>>(impl->d_hdrBloom_buffer, impl->d_convergence_buffer,
                                                               impl->WIDTH, impl->HEIGHT, impl->d_value);
        cudaMemcpy(&impl->value, impl->d_value, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(impl->d_convergence_buffer, impl->d_hdrBloom_buffer, impl->BUFFER_SIZE_FLOAT3,
                   cudaMemcpyDeviceToDevice);
    }
}

void Renderer::clean_up()
{
    cudaEventDestroy(impl->event_start);
    cudaEventDestroy(impl->event_stop);
    impl->radiance_pass.destroy();

    cudaFree(impl->d_isInside);
    cudaFree(impl->d_lastBounceWasDelta);
    cudaFree(impl->d_lastBsdfPdf);
    cudaFree(impl->d_origins);
    cudaFree(impl->d_directions);
    cudaFree(impl->d_throughputs);
    cudaFree(impl->d_rng);
    cudaFree(impl->d_accum_buffer);
    cudaFree(impl->d_value);

    cudaFree(impl->d_normalized_buffer);

    cudaFree(impl->d_bloom_buffer);
    cudaFree(impl->d_hdrBloom_buffer);
    cudaFree(impl->d_temp_buffer);

    cudaFree(impl->d_convergence_buffer);

    cudaFree(impl->d_lvl1);
    cudaFree(impl->d_lvl2);

    cudaStreamDestroy(impl->stream);
}

float3 *Renderer::get_finalized_image()
{
    if (!impl->d_normalized_buffer)
    {
        std::cerr << "Error: d_normalized_buffer is null" << std::endl;
        return nullptr;
    }

    // Allocate CPU memory for the image
    float3 *h_image = (float3 *)malloc(impl->BUFFER_SIZE_FLOAT3);

    if (!h_image)
    {
        std::cerr << "Error: Failed to allocate CPU memory for finalized image" << std::endl;
        return nullptr;
    }

    // Copy GPU buffer to CPU
    cudaError_t err = cudaMemcpy(h_image, impl->d_normalized_buffer, impl->BUFFER_SIZE_FLOAT3, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        std::cerr << "Error: cudaMemcpy failed - " << cudaGetErrorString(err) << std::endl;
        free(h_image);
        return nullptr;
    }

    return h_image;
}