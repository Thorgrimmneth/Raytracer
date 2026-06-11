#include "mesh_loader.cuh"

HOST 
MeshAndPrimitive loadTriangleMesh(const std::string &p_path, int materialIndex, int index, float3 scale,
                                       Quaternion rotation, float3 translation)
{
    std::cout << "Loading: " << p_path << std::endl;

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
    float3 mini = make_float3(+INFINITY);
    float3 maxi = make_float3(-INFINITY);
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

        // Add vertices, normals, and UVs
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
        {
            float3 vertex = make_float3(mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z);
            vertex = transformPoint(vertex, scale, rotation, translation);

            mini = getMin(mini, vertex);
            maxi = getMax(maxi, vertex);
            vertices.push_back(make_float3(vertex.x, vertex.y, vertex.z));

            float3 normal = make_float3(mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z);

            normal = transformNormal(normal, rotation);

            normals.push_back(make_float3(normal.x, normal.y, normal.z));

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
            float area =
                0.5f * abs((vertices[tri.y].x - vertices[tri.x].x) * (vertices[tri.z].y - vertices[tri.x].y) -
                           (vertices[tri.y].y - vertices[tri.x].y) * (vertices[tri.z].x - vertices[tri.x].x));
            totalArea += area;
            areaCdf.push_back(area);
            triangles.push_back(tri);
        }

        cptTriangles += mesh->mNumFaces;
        cptVertices += mesh->mNumVertices;

        std::cout << "-- [DONE] " << mesh->mNumFaces << " triangles, " << mesh->mNumVertices << " vertices."
                  << std::endl;
    }

    std::cout << "[DONE] " << scene->mNumMeshes << " meshes, " << cptTriangles << " triangles, " << cptVertices
              << " vertices." << std::endl;
    // Create TriangleMesh structure
    TriangleMesh triMesh;

    // Basic mesh metadata
    triMesh.triangleCount = static_cast<int>(triangles.size());
    triMesh.vertexCount = static_cast<int>(vertices.size());
    triMesh.materialIndex = materialIndex;

    // GPU pointers
    triMesh.triangles = nullptr;
    triMesh.vertices = nullptr;
    triMesh.normals = nullptr;
    triMesh.uvs = nullptr;
    triMesh.triangleAreaCdf = nullptr;

    // Sampling data
    triMesh.meshArea = totalArea;

    // Allocate and copy triangles to GPU
    if (!triangles.empty())
    {
        cudaMalloc(&triMesh.triangles, triangles.size() * sizeof(uint3));

        cudaMemcpy(triMesh.triangles, triangles.data(), triangles.size() * sizeof(uint3),
                   cudaMemcpyHostToDevice);
    }

    // Allocate and copy vertices to GPU
    if (!vertices.empty())
    {
        cudaMalloc(&triMesh.vertices, vertices.size() * sizeof(float3));

        cudaMemcpy(triMesh.vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy normals to GPU
    if (!normals.empty())
    {
        cudaMalloc(&triMesh.normals, normals.size() * sizeof(float3));

        cudaMemcpy(triMesh.normals, normals.data(), normals.size() * sizeof(float3), cudaMemcpyHostToDevice);
    }

    // Allocate and copy UVs to GPU
    if (!uvs.empty())
    {
        cudaMalloc(&triMesh.uvs, uvs.size() * sizeof(float2));

        cudaMemcpy(triMesh.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    }

    // Allocate and copy triangle area CDF to GPU
    if (!areaCdf.empty())
    {
        cudaMalloc(&triMesh.triangleAreaCdf, areaCdf.size() * sizeof(float));

        cudaMemcpy(triMesh.triangleAreaCdf, areaCdf.data(), areaCdf.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    triMesh.meshArea = totalArea;

    return MeshAndPrimitive(triMesh, mini, maxi, ObjectType::TRIANGLE, index);
}