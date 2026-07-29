#include "texture.hpp"
#include "io/stb_include.hpp"

void Texture::saveJPG(const std::string &p_path, const int p_quality)
{
    stbi_write_jpg(p_path.c_str(), int(_width), _height, _nbChannels, _pixels.data(), p_quality);
}

void Texture::savePNG(const std::string &p_path)
{
    stbi_write_png(p_path.c_str(), int(_width), _height, _nbChannels, _pixels.data(), 0);
}