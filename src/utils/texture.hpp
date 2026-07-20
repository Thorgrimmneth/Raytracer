#ifndef __RT_ISICG_IMAGE__
#define __RT_ISICG_IMAGE__

#include "utils/defines_cpu.hpp"
#include "renderingGPU/utils/op.cuh"
#include <string>
#include <vector>

class Texture {
  public:
    Texture() = default;
    Texture(const int p_width, const int p_height, const int _nbChannels = 3)
        : _width(p_width), _height(p_height), _pixels(_width * _height * _nbChannels, 0)
    {
        _pixels.shrink_to_fit();
    }
    ~Texture() = default;

    inline const int getWidth() const { return _width; }
    inline const int getHeight() const { return _height; }
    inline std::vector<unsigned char> &getPixels() { return _pixels; }
    inline const std::vector<unsigned char> &getPixels() const { return _pixels; }

    void setPixel(const int p_i, const int p_j, const float3 &p_color)
    {
        const int pixelId = (p_i + p_j * _width) * _nbChannels;
        _pixels[pixelId] = static_cast<unsigned char>(p_color.x * 255);
        _pixels[pixelId + 1] = static_cast<unsigned char>(p_color.y * 255);
        _pixels[pixelId + 2] = static_cast<unsigned char>(p_color.z * 255);
    }

    void setPixel(const int p_i, const int p_j, const float4 &p_color)
    {
        const int pixelId = (p_i + p_j * _width) * _nbChannels;
        _pixels[pixelId] = static_cast<unsigned char>(p_color.x * 255);
        _pixels[pixelId + 1] = static_cast<unsigned char>(p_color.y * 255);
        _pixels[pixelId + 2] = static_cast<unsigned char>(p_color.z * 255);
        _pixels[pixelId + 3] = static_cast<unsigned char>(p_color.w * 255);
    }

    void saveJPG(const std::string &p_path, const int p_quality = 100);

    void createFromRaw(unsigned char *p_img, int width, int height)
    {
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                int index = (j * width + i) * 3;
                float3 color = make_float3(p_img[index] / 255.f, p_img[index + 1] / 255.f, p_img[index + 2] / 255.f);
                setPixel(i, j, color);
            }
        }
    }

  private:
    int _nbChannels = 3;
    int _width;
    int _height;

    std::vector<unsigned char> _pixels;
};

#endif // __RT_ISICG_IMAGE__
