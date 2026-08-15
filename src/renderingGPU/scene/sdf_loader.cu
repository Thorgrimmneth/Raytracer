#include "sdf_loader.cuh"

SDF load_sdf(const std::string& filename)
{
    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file)
        throw std::runtime_error(
            "Cannot open SDF file: " + filename
        );


    // --------------------------------
    // Magic
    // --------------------------------

    char magic[4];

    file.read(
        magic,
        sizeof(magic)
    );

    if (!file)
        throw std::runtime_error(
            "Failed to read SDF header"
        );

    if (std::string(magic, 4) != "SDF1")
        throw std::runtime_error(
            "Invalid SDF file"
        );


    // --------------------------------
    // Resolution
    // --------------------------------

    uint32_t resolution;

    file.read(
        reinterpret_cast<char*>(&resolution),
        sizeof(resolution)
    );

    if (!file)
        throw std::runtime_error(
            "Failed to read SDF resolution"
        );


    // --------------------------------
    // Allocate grid
    // --------------------------------

    SignedGrid grid;

    grid.resolution = resolution;

    size_t voxel_count =
        static_cast<size_t>(resolution) *
        static_cast<size_t>(resolution) *
        static_cast<size_t>(resolution);

    grid.data = new float[voxel_count];


    // --------------------------------
    // Read SDF data
    // --------------------------------

    file.read(
        reinterpret_cast<char*>(grid.data),
        voxel_count * sizeof(float)
    );

    if (!file)
    {
        delete[] grid.data;

        throw std::runtime_error(
            "Error reading SDF data"
        );
    }


    // --------------------------------
    // Create SDF
    // --------------------------------

    SDF sdf;
    sdf.type = SDFType::SignedGrid;

    sdf.signedGrid = grid;

    printf("Loaded SDF from %s with resolution %u\n", filename.c_str(), resolution);
    printf("Size : %zu bytes = %zu mb\n", voxel_count * sizeof(float), voxel_count * sizeof(float) / (1024 * 1024));
    return sdf;
}