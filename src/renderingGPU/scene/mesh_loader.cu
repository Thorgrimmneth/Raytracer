#include "mesh_loader.cuh"

// Internal helper: Load geometry without transform
HOST MeshGeometry loadMeshGeometry(const std::string &p_path)
{
    std::cout << "Loading geometry: " << p_path << std::endl;

    Assimp::Importer importer;

    // Read scene and triangulate meshes
    const aiScene *const scene =
        importer.ReadFile(p_path, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_GenUVCoords);

    if (scene == nullptr)
    {
        throw std::runtime_error("Failed to load file: " + p_path);
    }

    // Aggregate all meshes into one
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint3> triangles;

    unsigned int cptTriangles = 0;
    unsigned int cptVertices = 0;
    float totalArea = 0.f;
    std::vector<float> areaCdf;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh *const mesh = scene->mMeshes[m];
        if (mesh == nullptr)
        {
            throw std::runtime_error("Failed to load file: " + p_path + ": mesh is null");
        }

        std::cout << "-- Load mesh " << m + 1 << "/" << scene->mNumMeshes << std::endl;

        const bool hasUV = mesh->HasTextureCoords(0);
        int vertexOffset = vertices.size();

        // Add vertices, normals, and UVs (no transform applied)
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
        {
            float3 vertex = make_float3(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
            vertices.push_back(vertex);

            float3 normal = make_float3(mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z);
            normals.push_back(normalize(normal));

            if (hasUV)
            {
                uvs.push_back(make_float2(mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y));
            }
            else
            {
                uvs.push_back(make_float2(0.f, 0.f));
            }
        }

        // Add triangles
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
        {
            const aiFace &face = mesh->mFaces[f];

            uint3 tri;
            tri.x = vertexOffset + face.mIndices[0];
            tri.y = vertexOffset + face.mIndices[1];
            tri.z = vertexOffset + face.mIndices[2];

            float3 e1 = vertices[tri.y] - vertices[tri.x];
            float3 e2 = vertices[tri.z] - vertices[tri.x];

            float area = 0.5f * length(cross(e1, e2));

            if (area <= 1e-8f)
                continue;

            totalArea += area;

            // cumulative CDF
            areaCdf.push_back(totalArea);

            triangles.push_back(tri);
        }
        for (float &v : areaCdf)
            v /= totalArea;

        cptTriangles += mesh->mNumFaces;
        cptVertices += mesh->mNumVertices;

        std::cout << "-- [DONE] " << mesh->mNumFaces << " triangles, " << mesh->mNumVertices << " vertices."
                  << std::endl;
    }

    std::cout << "[DONE] " << scene->mNumMeshes << " meshes, " << cptTriangles << " triangles, " << cptVertices
              << " vertices." << std::endl;

    // Create MeshGeometry structure
    MeshGeometry geometry;

    geometry.triangleCount = static_cast<int>(triangles.size());
    geometry.vertexCount = static_cast<int>(vertices.size());
    geometry.meshArea = totalArea;

    // Allocate and copy triangles to GPU
    if (!triangles.empty())
    {
        cudaMalloc(&geometry.triangles, triangles.size() * sizeof(uint3));
        cudaMemcpy(geometry.triangles, triangles.data(), triangles.size() * sizeof(uint3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy vertices to GPU
    if (!vertices.empty())
    {
        cudaMalloc(&geometry.vertices, vertices.size() * sizeof(float3));
        cudaMemcpy(geometry.vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy normals to GPU
    if (!normals.empty())
    {
        cudaMalloc(&geometry.normals, normals.size() * sizeof(float3));
        cudaMemcpy(geometry.normals, normals.data(), normals.size() * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy UVs to GPU
    if (!uvs.empty())
    {
        cudaMalloc(&geometry.uvs, uvs.size() * sizeof(float2));
        cudaMemcpy(geometry.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    }

    // Allocate and copy triangle area CDF to GPU
    if (!areaCdf.empty())
    {
        cudaMalloc(&geometry.triangleAreaCdf, areaCdf.size() * sizeof(float));
        cudaMemcpy(geometry.triangleAreaCdf, areaCdf.data(), areaCdf.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    return geometry;
}

// Internal helper: Load geometry without transform
HOST MeshGeometry loadMeshGeometry(const std::string &p_path, const float3 scale)
{
    std::cout << "Loading geometry: " << p_path << std::endl;

    Assimp::Importer importer;

    // Read scene and triangulate meshes
    const aiScene *const scene =
        importer.ReadFile(p_path, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_GenUVCoords);

    if (scene == nullptr)
    {
        throw std::runtime_error("Failed to load file: " + p_path);
    }

    // Aggregate all meshes into one
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<uint3> triangles;

    unsigned int cptTriangles = 0;
    unsigned int cptVertices = 0;
    float totalArea = 0.f;
    std::vector<float> areaCdf;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh *const mesh = scene->mMeshes[m];
        if (mesh == nullptr)
        {
            throw std::runtime_error("Failed to load file: " + p_path + ": mesh is null");
        }

        std::cout << "-- Load mesh " << m + 1 << "/" << scene->mNumMeshes << std::endl;

        const bool hasUV = mesh->HasTextureCoords(0);
        int vertexOffset = vertices.size();

        // Add vertices, normals, and UVs (apply scaling)
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
        {
            float3 vertex = make_float3(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z) * scale;
            vertices.push_back(vertex);

            float3 normal = make_float3(mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z);
            normals.push_back(normalize(normal));

            if (hasUV)
            {
                uvs.push_back(make_float2(mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y));
            }
            else
            {
                uvs.push_back(make_float2(0.f, 0.f));
            }
        }

        // Add triangles
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
        {
            const aiFace &face = mesh->mFaces[f];

            uint3 tri;
            tri.x = vertexOffset + face.mIndices[0];
            tri.y = vertexOffset + face.mIndices[1];
            tri.z = vertexOffset + face.mIndices[2];

            float3 e1 = vertices[tri.y] - vertices[tri.x];
            float3 e2 = vertices[tri.z] - vertices[tri.x];

            float area = 0.5f * length(cross(e1, e2));

            if (area <= 1e-8f)
                continue;

            totalArea += area;

            // cumulative CDF
            areaCdf.push_back(totalArea);

            triangles.push_back(tri);
        }
        for (float &v : areaCdf)
            v /= totalArea;

        cptTriangles += mesh->mNumFaces;
        cptVertices += mesh->mNumVertices;

        std::cout << "-- [DONE] " << mesh->mNumFaces << " triangles, " << mesh->mNumVertices << " vertices."
                  << std::endl;
    }

    std::cout << "[DONE] " << scene->mNumMeshes << " meshes, " << cptTriangles << " triangles, " << cptVertices
              << " vertices." << std::endl;

    // Create MeshGeometry structure
    MeshGeometry geometry;

    geometry.triangleCount = static_cast<int>(triangles.size());
    geometry.vertexCount = static_cast<int>(vertices.size());
    geometry.meshArea = totalArea;

    // Allocate and copy triangles to GPU
    if (!triangles.empty())
    {
        cudaMalloc(&geometry.triangles, triangles.size() * sizeof(uint3));
        cudaMemcpy(geometry.triangles, triangles.data(), triangles.size() * sizeof(uint3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy vertices to GPU
    if (!vertices.empty())
    {
        cudaMalloc(&geometry.vertices, vertices.size() * sizeof(float3));
        cudaMemcpy(geometry.vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy normals to GPU
    if (!normals.empty())
    {
        cudaMalloc(&geometry.normals, normals.size() * sizeof(float3));
        cudaMemcpy(geometry.normals, normals.data(), normals.size() * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy UVs to GPU
    if (!uvs.empty())
    {
        cudaMalloc(&geometry.uvs, uvs.size() * sizeof(float2));
        cudaMemcpy(geometry.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    }

    // Allocate and copy triangle area CDF to GPU
    if (!areaCdf.empty())
    {
        cudaMalloc(&geometry.triangleAreaCdf, areaCdf.size() * sizeof(float));
        cudaMemcpy(geometry.triangleAreaCdf, areaCdf.data(), areaCdf.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    return geometry;
}

// Helper: Create 3x4 transformation matrix from scale, rotation quaternion, and translation
HOST void buildTransformMatrix(float *out_transform, float3 scale, Quaternion rotation, float3 translation)
{
    // Convert quaternion to rotation matrix (3x3)
    float q_x = rotation.x;
    float q_y = rotation.y;
    float q_z = rotation.z;
    float q_w = rotation.w;

    // Rotation matrix from quaternion
    float r00 = 1.f - 2.f * (q_y * q_y + q_z * q_z);
    float r01 = 2.f * (q_x * q_y - q_w * q_z);
    float r02 = 2.f * (q_x * q_z + q_w * q_y);

    float r10 = 2.f * (q_x * q_y + q_w * q_z);
    float r11 = 1.f - 2.f * (q_x * q_x + q_z * q_z);
    float r12 = 2.f * (q_y * q_z - q_w * q_x);

    float r20 = 2.f * (q_x * q_z - q_w * q_y);
    float r21 = 2.f * (q_y * q_z + q_w * q_x);
    float r22 = 1.f - 2.f * (q_x * q_x + q_y * q_y);

    // Build 3x4 matrix: [R*S | T]
    // Row 0: [r00*sx, r01*sx, r02*sx, tx]
    out_transform[0] = r00 * scale.x;
    out_transform[1] = r01 * scale.x;
    out_transform[2] = r02 * scale.x;
    out_transform[3] = translation.x;

    // Row 1: [r10*sy, r11*sy, r12*sy, ty]
    out_transform[4] = r10 * scale.y;
    out_transform[5] = r11 * scale.y;
    out_transform[6] = r12 * scale.y;
    out_transform[7] = translation.y;

    // Row 2: [r20*sz, r21*sz, r22*sz, tz]
    out_transform[8] = r20 * scale.z;
    out_transform[9] = r21 * scale.z;
    out_transform[10] = r22 * scale.z;
    out_transform[11] = translation.z;
}

// Create an instance from a loaded geometry
HOST MeshInstance createMeshInstance(CudaSceneHelper &sceneHelper, int geometryIndex, int materialIndex, float3 scale, Quaternion rotation,
                                     float3 translation)
{
    MeshInstance instance;
    instance.geometryIndex = geometryIndex;
    instance.materialIndex = materialIndex;
    instance.worldArea = sceneHelper.meshGeometriesGPU[geometryIndex].meshArea * scale.x * scale.x;

    buildTransformMatrix(instance.transform, scale, rotation, translation);
    return instance;
}

MeshGeometry PlaneToMesh(const Plane &plane, float size)
{
    MeshGeometry mesh;

    float3 n = make_float3(plane.normal.x, plane.normal.y, plane.normal.z);

    float d = plane.normal.w;

    float3 pos = -d * n;

    mesh.vertexCount = 4;
    mesh.triangleCount = 2;

    mesh.vertices = new float3[4];
    mesh.normals = new float3[4];
    mesh.uvs = new float2[4];
    mesh.triangles = new uint3[2];

    float3 tangent =
        fabs(n.y) < 0.999f ? normalize(cross(make_float3(0, 1, 0), n)) : normalize(cross(make_float3(1, 0, 0), n));

    float3 bitangent = cross(n, tangent);

    float h = size * 0.5f;

    mesh.vertices[0] = pos + (-tangent - bitangent) * h;
    mesh.vertices[1] = pos + (tangent - bitangent) * h;
    mesh.vertices[2] = pos + (tangent + bitangent) * h;
    mesh.vertices[3] = pos + (-tangent + bitangent) * h;

    for (int i = 0; i < 4; ++i)
        mesh.normals[i] = n;

    mesh.uvs[0] = make_float2(0, 0);
    mesh.uvs[1] = make_float2(1, 0);
    mesh.uvs[2] = make_float2(1, 1);
    mesh.uvs[3] = make_float2(0, 1);

    mesh.triangles[0] = make_uint3(0, 1, 2);
    mesh.triangles[1] = make_uint3(0, 2, 3);

    mesh.meshArea = size * size;
    mesh.triangleAreaCdf = new float[2];
    mesh.triangleAreaCdf[0] = 0.5f;
    mesh.triangleAreaCdf[1] = 1.0f;

    return mesh;
}