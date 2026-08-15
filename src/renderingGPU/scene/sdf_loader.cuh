#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <cstdint>
#include <stdexcept>

#include "../objects/signed_grid.cuh"
#include "../objects/sdf.cuh"


SDF load_sdf(const std::string& filename);