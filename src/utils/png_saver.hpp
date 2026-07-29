#include <png.h>
#include <cstdio>
#include <vector>

bool writePNG(
    const char* filename,
    const unsigned char* image,
    int width,
    int height)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp)
        return false;

    png_structp png =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png)
    {
        fclose(fp);
        return false;
    }

    png_infop info = png_create_info_struct(png);

    if (!info)
    {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);

    // Indique que les pixels sont encodés en sRGB.
    png_set_sRGB(png, info, PNG_sRGB_INTENT_PERCEPTUAL);

    png_write_info(png, info);

    std::vector<png_bytep> rows(height);

    for (int y = 0; y < height; ++y)
        rows[y] = (png_bytep)(image + y * width * 3);

    png_write_image(png, rows.data());

    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);

    fclose(fp);

    return true;
}