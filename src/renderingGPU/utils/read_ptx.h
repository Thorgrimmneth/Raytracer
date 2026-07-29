#pragma once

#include <fstream>
#include <string>
#include <stdexcept>

std::string loadTextFile(
    const std::string& path
)
{
    std::ifstream file(path);

    if(!file)
    {
        throw std::runtime_error(
            "Cannot open " + path
        );
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}