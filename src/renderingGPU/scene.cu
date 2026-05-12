#include "lights/light.cuh"
#include "scene.cuh"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "objectsUtils/bvh.cuh"

__device__ bool CudaScene::intersect(const Ray &p_ray, const float p_tMin, const float p_tMax, HitRecord &p_hitRecord) const
{
	float tMax = p_tMax;
	bool hit = false;
	if (bvhScene.intersect(p_ray, p_tMin, tMax, p_hitRecord))
	{
		tMax = p_hitRecord.distance; // update tMax to conserve the nearest hit
		hit = true;
	}
	return hit;
}

__device__ bool CudaScene::intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const
{
	return bvhScene.intersectAny(p_ray, p_tMin, p_tMax, materials);
}

__host__
void CudaScene::uploadObjects(
    std::vector<Sphere> spheresGPU,
    std::vector<Plane> planesGPU,
    std::vector<TriangleMesh> triangleMeshesGPU,
    std::vector<BaseObject> primitivesGPU,
    std::vector<ImplicitSphere> implicitSpheresGPU)
{
    // =========================
    // Upload primitives
    // =========================
    int nbObjects = primitivesGPU.size();

    if (nbObjects > 0)
    {
        cudaMalloc(&primitives, nbObjects * sizeof(BaseObject));
        cudaMemcpy(primitives,
                   primitivesGPU.data(),
                   nbObjects * sizeof(BaseObject),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        primitives = nullptr;
    }

    // =========================
    // Build BVH
    // =========================
    bvhScene = BVHScene::buildBVHScene(&primitivesGPU,
                                       &spheresGPU,
                                       &planesGPU,
                                       &triangleMeshesGPU,
                                       &implicitSpheresGPU);

    // =========================
    // Upload spheres
    // =========================
    nbSpheres = spheresGPU.size();
    if (nbSpheres > 0)
    {
        cudaMalloc(&spheres, nbSpheres * sizeof(Sphere));
        cudaMemcpy(spheres,
                   spheresGPU.data(),
                   nbSpheres * sizeof(Sphere),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        spheres = nullptr;
    }

    // =========================
    // Upload planes
    // =========================
    nbPlanes = planesGPU.size();
    if (nbPlanes > 0)
    {
        cudaMalloc(&planes, nbPlanes * sizeof(Plane));
        cudaMemcpy(planes,
                   planesGPU.data(),
                   nbPlanes * sizeof(Plane),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        planes = nullptr;
    }

    // =========================
    // Upload meshes
    // =========================
    nbTriangleMeshes = triangleMeshesGPU.size();
    if (nbTriangleMeshes > 0)
    {
        cudaMalloc(&triangleMeshes,
                   nbTriangleMeshes * sizeof(TriangleMesh));

        cudaMemcpy(triangleMeshes,
                   triangleMeshesGPU.data(),
                   nbTriangleMeshes * sizeof(TriangleMesh),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        triangleMeshes = nullptr;
    }
               
    nbImplicitSpheres = implicitSpheresGPU.size();
    if (nbImplicitSpheres > 0)
    {
        cudaMalloc(&implicitSpheres,
                   nbImplicitSpheres * sizeof(ImplicitSphere));
        cudaMemcpy(implicitSpheres,
                   implicitSpheresGPU.data(),
                   nbImplicitSpheres * sizeof(ImplicitSphere),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        implicitSpheres = nullptr;
    }

    bvhScene.d_primitives = primitives;
    bvhScene.d_spheres    = spheres;
    bvhScene.d_planes     = planes;
    bvhScene.d_meshes     = triangleMeshes;
    bvhScene.d_implicitSpheres = implicitSpheres;
}

__host__
void CudaScene::uploadLights(
                             std::vector<Light> lightsGPU){
    nbLights = lightsGPU.size();
    if (nbLights > 0)
    {
        cudaMalloc(&lights,
                   nbLights * sizeof(Light));

        cudaMemcpy(lights,
                   lightsGPU.data(),
                   nbLights * sizeof(Light),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        lights = nullptr;
    }
}

__host__ 
void CudaScene::uploadMaterials(std::vector<Material> materialsGPU)
{
	nbMaterials = materialsGPU.size();

	if (nbMaterials > 0)
	{
		cudaMalloc(&materials,
				   nbMaterials * sizeof(Material));

		cudaMemcpy(materials,
				   materialsGPU.data(),
				   nbMaterials * sizeof(Material),
				   cudaMemcpyHostToDevice);
	}
	else
	{
		materials = nullptr;
	}
}

__device__
float CudaScene::lightPdf(
    const float3& origin,
    const float3& dir) const
{
    Ray ray(origin, dir);

    HitRecord hit;

    if(!intersect(ray, 1e-4f, 1e30f, hit))
        return 0.f;

    const Material& mtl =
        materials[hit.materialIndex];

    if(mtl.type() != MaterialType::EMISSIVE)
        return 0.f;

    float pdf = 0.f;

    // sphere emissive
    if(hit.objectType == HIT_SPHERE || hit.objectType == HIT_SPHERE_IMPLICIT)
    {
        float3 lightCenter;
        float radius;

        if (hit.objectType == HIT_SPHERE)
        {
            const Sphere& s = spheres[hit.objectIndex];
            lightCenter = s.center1;
            radius = s.radius;
        }
        else
        {
            const ImplicitSphere& s = implicitSpheres[hit.objectIndex];
            lightCenter = s.center1;
            radius = s.radius;
        }

        float dist2 =
            length2(hit.point - origin);

        float3 n =
            normalize(hit.point - lightCenter);

        float cosTheta =
            max(dot(n, -dir), 0.f);

        if(cosTheta <= 0.f)
            return 0.f;

        float area =
            4.f * GPUPIf * radius * radius;

        float pdfArea = 1.f / area;

        pdf =
            pdfArea * dist2 / cosTheta;
    }

    // triangle mesh emissive
    else if(hit.objectType == HIT_TRIANGLE_MESH)
    {
        const TriangleMesh& mesh =
            triangleMeshes[hit.objectIndex];

        float3 n = hit.normal;

        float dist2 =
            length2(hit.point - origin);

        float cosTheta =
            max(dot(n, -dir), 0.f);

        if(cosTheta <= 0.f)
            return 0.f;

        float pdfArea =
            1.f / mesh.meshArea;

        pdf =
            pdfArea * dist2 / cosTheta;
    }

    // lumière choisie uniformément
    pdf *= (1.f / nbLights);

    return pdf;
};

void sceneSize(CudaScene gpuScene, std::vector<BaseObject> primitivesGPU)
{
    printf("Size of one BVH node: %zu bytes\n", sizeof(BVHSceneNode));
    printf("Size of AABB: %zu bytes\n", sizeof(AABB));
    printf("Size of BaseObject: %zu bytes\n", sizeof(BaseObject));
    printf("Size of Sphere: %zu bytes\n", sizeof(Sphere));
    printf("Size of Plane: %zu bytes\n", sizeof(Plane));
    printf("Size of Material: %zu bytes\n", sizeof(Material));
    printf("Size of Light: %zu bytes\n", sizeof(Light));
    printf("Size of ImplicitSphere: %zu bytes\n", sizeof(ImplicitSphere));
    printf("\n");
    size_t totalSize = 0;
    totalSize += gpuScene.nbSpheres * sizeof(Sphere);
    printf("Size of spheres: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbSpheres * sizeof(Sphere), (1.f - (gpuScene.nbSpheres * sizeof(Sphere) / 38240.f)) * 100.f);
    totalSize += gpuScene.nbPlanes * sizeof(Plane);
    printf("Size of planes: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbPlanes * sizeof(Plane), (1.f - (gpuScene.nbPlanes * sizeof(Plane) / 20.f))* 100.f);
    totalSize += gpuScene.nbTriangleMeshes * sizeof(TriangleMesh);
    printf("Size of triangle meshes: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbTriangleMeshes * sizeof(TriangleMesh), (1.f - (gpuScene.nbTriangleMeshes * sizeof(TriangleMesh) / 1.f)) * 100.f);
    totalSize += gpuScene.nbMaterials * sizeof(Material);
    printf("Size of materials: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbMaterials * sizeof(Material), (1.f - (gpuScene.nbMaterials * sizeof(Material) / 15360.f)) * 100.f);
    totalSize += gpuScene.nbLights * sizeof(Light);
    printf("Size of lights: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbLights * sizeof(Light), (1.f - (gpuScene.nbLights * sizeof(Light) / 192.f)) * 100.f);
    totalSize += primitivesGPU.size() * sizeof(BaseObject);
    printf("Size of primitives: %zu bytes. %2.2f gain compared to v1\n", primitivesGPU.size() * sizeof(BaseObject), (1.f - (primitivesGPU.size() * sizeof(BaseObject) / 22992.f)) * 100.f);
    totalSize += gpuScene.bvhScene.getDeviceSize();
    printf("Size of implicit spheres: %zu bytes. %2.2f gain compared to v1\n", gpuScene.nbImplicitSpheres * sizeof(ImplicitSphere), (1.f - (gpuScene.nbImplicitSpheres * sizeof(ImplicitSphere) / 400.f)) * 100.f);
    totalSize += gpuScene.nbImplicitSpheres * sizeof(ImplicitSphere);
    printf("BVH size: %zu bytes. %2.2f gain compared to v1\n", gpuScene.bvhScene.getDeviceSize(), (1.f - (gpuScene.bvhScene.getDeviceSize() / 68928.f)) * 100.f);
    printf("Total size of GPU data: %zu bytes. %2.2f gain compared to v1\n", totalSize, (1.f - (totalSize / 145732.f)) * 100.f);
}

CudaScene spheresScene(float4 sunDir)
{
    CudaScene gpuScene;

    std::vector<Sphere> spheresGPU;
    std::vector<Plane> planesGPU;
    std::vector<TriangleMesh> triangleMeshesGPU;
    std::vector<float3> verticesGPU;
    std::vector<BaseObject> primitivesGPU;
    std::vector<Material> materialsGPU;
    std::vector<Light> lightsGPU;

    // ===== PLAN =====
    {
        Plane p;
        p.delta = 0.f;
        p.normal = make_float3(0.f, 1.f, 0.f);

        Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
        materialsGPU.push_back(ground);
        p.materialIndex = materialsGPU.size() - 1;

        primitivesGPU.push_back(BaseObject{make_float3(-1e3f, 0.f, -1e3f), make_float3(1e3f, 0.f, 1e3f),ObjectType::PLANE, (int)planesGPU.size()});
        planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material mirror      = Material::makeMaterial(make_float3(0.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(1.f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive    = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);

    int mirrorIdx = materialsGPU.size(); materialsGPU.push_back(mirror);
    int transparentIdx = materialsGPU.size(); materialsGPU.push_back(transparent);
    int emissiveIdx = materialsGPU.size(); materialsGPU.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 11;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
        {
            double choose_mat = RT::randomDouble();

            float3 center = make_float3(
                i + 0.9f * RT::randomFloat(),
                0.2f,
                j + 0.9f * RT::randomFloat()
            );

            if (
                length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            Sphere s;
            s.center1 = center;
            s.center2 = center;
            s.radius = 0.2f;

            // ===== MATERIAL =====
            if (choose_mat < 0.6)
            {
                Material mat = Material::makeMaterial(
                    make_float3(
                        RT::randomFloat(),
                        RT::randomFloat(),
                        RT::randomFloat()
                    ),
                    LAMBERT,
                    1.0f
                );
                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.8)
            {
                Material mat = Material::randomMetal();
                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.88)
            {
                s.materialIndex = mirrorIdx;
            }
            else
            {
                s.materialIndex = transparentIdx;
            }

            // ===== AABB =====
            float3 r = make_float3(s.radius);

            primitivesGPU.push_back(BaseObject{
                center - r,
                center + r,
                ObjectType::SPHERE,
                (int)spheresGPU.size()
            });
            spheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex)
    {
        Sphere s;
        s.center1 = center;
        s.center2 = center;
        s.radius = radius;
        s.materialIndex = matIndex;

        float3 r = make_float3(radius);

        primitivesGPU.push_back(BaseObject{
            center - r,
            center + r,
            ObjectType::SPHERE,
            (int)spheresGPU.size()
        });

        spheresGPU.push_back(s);
		Material& mat = materialsGPU[matIndex];

		if (mat.type() == MaterialType::EMISSIVE)
		{
			Light l;
			l.metadata = Light::packMetadata(LightType::SPHERE_GEOM, spheresGPU.size() - 1);
			lightsGPU.push_back(l);
		}
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx);

    // ===== LIGHT (SUN) =====
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(spheresGPU, planesGPU, triangleMeshesGPU, primitivesGPU, std::vector<ImplicitSphere>());
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);

    sceneSize(gpuScene, primitivesGPU);
    return gpuScene;
}

CudaScene implicitSpheresScene(float4 sunDir)
{
    CudaScene gpuScene;

    std::vector<ImplicitSphere> implicitSpheresGPU;
    std::vector<Plane> planesGPU;
    std::vector<TriangleMesh> triangleMeshesGPU;
    std::vector<BaseObject> primitivesGPU;
    std::vector<Material> materialsGPU;
    std::vector<Light> lightsGPU;

    // ===== PLAN =====
    {
        Plane p;
        p.delta = 0.f;
        p.normal = make_float3(0.f, 1.f, 0.f);

        Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
        materialsGPU.push_back(ground);
        p.materialIndex = materialsGPU.size() - 1;

        primitivesGPU.push_back(BaseObject{make_float3(-1e3f, 0.f, -1e3f), make_float3(1e3f, 0.f, 1e3f),ObjectType::PLANE, (int)planesGPU.size()});
        planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material mirror      = Material::makeMaterial(make_float3(0.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(1.f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive    = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);

    int mirrorIdx = materialsGPU.size(); materialsGPU.push_back(mirror);
    int transparentIdx = materialsGPU.size(); materialsGPU.push_back(transparent);
    int emissiveIdx = materialsGPU.size(); materialsGPU.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 11;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
        {
            double choose_mat = RT::randomDouble();

            float3 center = make_float3(
                i + 0.9f * RT::randomFloat(),
                0.2f,
                j + 0.9f * RT::randomFloat()
            );

            if (
                length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            ImplicitSphere s;
            s.center1 = center;
            s.center2 = center;
            s.radius = 0.2f;

            // ===== MATERIAL =====
            if (choose_mat < 0.6)
            {
                Material mat = Material::makeMaterial(
                    make_float3(
                        RT::randomFloat(),
                        RT::randomFloat(),
                        RT::randomFloat()
                    ),
                    LAMBERT,
                    1.0f
                );
                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.8)
            {
                Material mat = Material::randomMetal();
                materialsGPU.push_back(mat);
                s.materialIndex = materialsGPU.size() - 1;
            }
            else if (choose_mat < 0.88)
            {
                s.materialIndex = mirrorIdx;
            }
            else
            {
                s.materialIndex = transparentIdx;
            }

            // ===== AABB =====
            float3 r = make_float3(s.radius);

            primitivesGPU.push_back(BaseObject{
                center - r,
                center + r,
                ObjectType::IMPLICIT_SPHERE,
                (int)implicitSpheresGPU.size()
            });
            implicitSpheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex)
    {
        ImplicitSphere s;
        s.center1 = center;
        s.center2 = center;
        s.radius = radius;
        s.materialIndex = matIndex;

        float3 r = make_float3(radius);

        primitivesGPU.push_back(BaseObject{
            center - r,
            center + r,
            ObjectType::IMPLICIT_SPHERE,
            (int)implicitSpheresGPU.size()
        });

        implicitSpheresGPU.push_back(s);
        Material& mat = materialsGPU[matIndex];

        if (mat.type() == MaterialType::EMISSIVE)
        {
            Light l;
            l.metadata = Light::packMetadata(LightType::IMPLICIT_SPHERE_GEOM, implicitSpheresGPU.size() - 1);
            lightsGPU.push_back(l);
        }
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx);
    // ===== LIGHT (SUN) =====
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(std::vector<Sphere>(), planesGPU, triangleMeshesGPU, primitivesGPU, implicitSpheresGPU);
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);

    sceneSize(gpuScene, primitivesGPU);
    return gpuScene;
}

CudaScene singleObject(float4 sunDir)
{
    CudaScene gpuScene;
    std::vector<TriangleMesh> triangleMeshesGPU;
    std::vector<BaseObject> primitivesGPU;
    std::vector<Material> materialsGPU;
    std::vector<Light> lightsGPU;
    Material mat = Material::makeMaterial(
                    make_float3(
                        RT::randomFloat(),
                        RT::randomFloat(),
                        RT::randomFloat()
                    ),
                    LAMBERT,
                    1.0f
                );
    materialsGPU.push_back(mat);
    MeshAndPrimitive meshAndPrim = loadTriangleMesh("../data/bunny/Bunny.obj", materialsGPU.size() - 1, triangleMeshesGPU.size());
    triangleMeshesGPU.push_back(meshAndPrim.mesh);
    primitivesGPU.push_back(meshAndPrim.prim);
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    lightsGPU.push_back(l);
    gpuScene.uploadObjects(std::vector<Sphere>(), std::vector<Plane>(), triangleMeshesGPU, primitivesGPU, std::vector<ImplicitSphere>());
    gpuScene.uploadLights(lightsGPU);
    gpuScene.uploadMaterials(materialsGPU);
    return gpuScene;
}



__host__
MeshAndPrimitive loadTriangleMesh(const std::string& p_path, int materialIndex, int index)
{
    std::cout << "Loading: " << p_path << std::endl;
    
    Assimp::Importer importer;
    
    // Read scene and triangulate meshes
    const aiScene* const scene = importer.ReadFile(
        p_path, 
        aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_GenUVCoords
    );
    
    if (scene == nullptr) {
        throw std::runtime_error("Failed to load file: " + p_path);
    }
    
    // Aggregate all meshes into one
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<TriangleMeshGeometry> triangles;
    
    unsigned int cptTriangles = 0;
    unsigned int cptVertices = 0;
    float3 mini = make_float3(+INFINITY);
    float3 maxi = make_float3(-INFINITY);
    float totalArea;
    std::vector<float> areaCdf;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* const mesh = scene->mMeshes[m];
        if (mesh == nullptr) {
            throw std::runtime_error("Failed to load file: " + p_path + ": mesh is null");
        }
        
        std::cout << "-- Load mesh " << m + 1 << "/" << scene->mNumMeshes << std::endl;
        
        const bool hasUV = mesh->HasTextureCoords(0);
        int vertexOffset = vertices.size();
        
        // Add vertices, normals, and UVs
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            float3 vertex = make_float3(
                mesh->mVertices[v].x,
                mesh->mVertices[v].y,
                mesh->mVertices[v].z
            );
            mini = getMin(mini, vertex);
            maxi = getMax(maxi, vertex);
            vertices.push_back(vertex);
            
            normals.push_back(make_float3(
                mesh->mNormals[v].x,
                mesh->mNormals[v].y,
                mesh->mNormals[v].z
            ));
            
            if (hasUV) {
                uvs.push_back(make_float2(
                    mesh->mTextureCoords[0][v].x,
                    mesh->mTextureCoords[0][v].y
                ));
            } else {
                uvs.push_back(make_float2(0.f, 0.f));
            }
        }
        
        // Add triangles
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            TriangleMeshGeometry tri;
            tri.i0 = vertexOffset + face.mIndices[0];
            tri.i1 = vertexOffset + face.mIndices[1];
            tri.i2 = vertexOffset + face.mIndices[2];
            float area = 0.5f * abs((vertices[tri.i1].x - vertices[tri.i0].x) * (vertices[tri.i2].y - vertices[tri.i0].y) - (vertices[tri.i1].y - vertices[tri.i0].y) * (vertices[tri.i2].x - vertices[tri.i0].x));
            totalArea += area;
            areaCdf.push_back(area);
            triangles.push_back(tri);
        }
        
        cptTriangles += mesh->mNumFaces;
        cptVertices += mesh->mNumVertices;
        
        std::cout << "-- [DONE] " << mesh->mNumFaces << " triangles, " << mesh->mNumVertices << " vertices." << std::endl;
    }
    
    std::cout << "[DONE] " << scene->mNumMeshes << " meshes, " << cptTriangles << " triangles, " << cptVertices << " vertices." << std::endl;
    
    // Create TriangleMesh structure
    TriangleMesh triMesh;
    triMesh.triangleCount = triangles.size();
    triMesh.vertexCount = vertices.size();
    triMesh.materialIndex = materialIndex;
    
    triMesh.bvhNodes = buildBVH(triangles.data(), triMesh.triangleCount, vertices.data(), normals.data(), uvs.data(), triMesh.bvhNodeCount);
    // Allocate and copy triangles to GPU
    cudaMalloc(&triMesh.triangles, triangles.size() * sizeof(TriangleMeshGeometry));
    cudaMemcpy(triMesh.triangles, triangles.data(), triangles.size() * sizeof(TriangleMeshGeometry), cudaMemcpyHostToDevice);
    
    // Allocate and copy vertices to GPU
    cudaMalloc(&triMesh.vertices, vertices.size() * sizeof(float3));
    cudaMemcpy(triMesh.vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    
    // Allocate and copy normals to GPU
    cudaMalloc(&triMesh.normals, normals.size() * sizeof(float3));
    cudaMemcpy(triMesh.normals, normals.data(), normals.size() * sizeof(float3), cudaMemcpyHostToDevice);
    
    // Allocate and copy UVs to GPU
    cudaMalloc(&triMesh.uvs, uvs.size() * sizeof(float2));
    cudaMemcpy(triMesh.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    
    cudaMalloc(&triMesh.triangleAreaCdf, areaCdf.size() * sizeof(float));
    cudaMemcpy(triMesh.triangleAreaCdf, areaCdf.data(), areaCdf.size() * sizeof(float), cudaMemcpyHostToDevice);

    triMesh.meshArea = totalArea;
    
    return MeshAndPrimitive(triMesh, mini, maxi, ObjectType::TRIANGLE, index);
}
