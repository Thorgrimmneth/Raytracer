
#include "scene.cuh"

#include "scene_helper.cuh"

HOST void CudaScene::uploadObjects(CudaSceneHelper &helper)
{
    // =========================
    // Upload primitives
    // =========================
    int nbObjects = helper.primitivesGPU.size();

    if (nbObjects > 0)
    {
        cudaMalloc(&primitives, nbObjects * sizeof(BaseObject));
        cudaMemcpy(primitives, helper.primitivesGPU.data(), nbObjects * sizeof(BaseObject), cudaMemcpyHostToDevice);
    }
    else
    {
        primitives = nullptr;
    }

    // =========================
    // Build BVH
    // =========================
    bvhScene = BVHScene::buildBVHScene(&helper.primitivesGPU, &helper.spheresGPU, &helper.triangleMeshesGPU,
                                       &helper.implicitSpheresGPU);

    // =========================
    // Upload spheres
    // =========================
    nbSpheres = helper.spheresGPU.size();
    if (nbSpheres > 0)
    {
        cudaMalloc(&spheres, nbSpheres * sizeof(Sphere));
        cudaMemcpy(spheres, helper.spheresGPU.data(), nbSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    }
    else
    {
        spheres = nullptr;
    }

    // =========================
    // Upload planes
    // =========================
    nbPlanes = helper.planesGPU.size();
    if (nbPlanes > 0)
    {
        cudaMalloc(&planes, nbPlanes * sizeof(Plane));
        cudaMemcpy(planes, helper.planesGPU.data(), nbPlanes * sizeof(Plane), cudaMemcpyHostToDevice);
    }
    else
    {
        planes = nullptr;
    }

    // =========================
    // Upload meshes
    // =========================
    nbTriangleMeshes = helper.triangleMeshesGPU.size();
    if (nbTriangleMeshes > 0)
    {
        cudaMalloc(&triangleMeshes, nbTriangleMeshes * sizeof(TriangleMesh));

        cudaMemcpy(triangleMeshes, helper.triangleMeshesGPU.data(), nbTriangleMeshes * sizeof(TriangleMesh),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        triangleMeshes = nullptr;
    }

    nbImplicitSpheres = helper.implicitSpheresGPU.size();
    if (nbImplicitSpheres > 0)
    {
        cudaMalloc(&implicitSpheres, nbImplicitSpheres * sizeof(ImplicitSphere));
        cudaMemcpy(implicitSpheres, helper.implicitSpheresGPU.data(), nbImplicitSpheres * sizeof(ImplicitSphere),
                   cudaMemcpyHostToDevice);
    }
    else
    {
        implicitSpheres = nullptr;
    }

    bvhScene.d_primitives = primitives;
    bvhScene.d_spheres = spheres;
    bvhScene.d_planes = planes;
    bvhScene.d_meshes = triangleMeshes;
    bvhScene.d_implicitSpheres = implicitSpheres;
}

HOST void CudaScene::uploadLights(CudaSceneHelper &helper)
{
    nbLights = helper.lightsGPU.size();
    if (nbLights > 0)
    {
        cudaMalloc(&lights, nbLights * sizeof(Light));

        cudaMemcpy(lights, helper.lightsGPU.data(), nbLights * sizeof(Light), cudaMemcpyHostToDevice);
    }
    else
    {
        lights = nullptr;
    }
}

HOST void CudaScene::uploadMaterials(CudaSceneHelper &helper)
{
    nbMaterials = helper.materialsGPU.size();

    if (nbMaterials > 0)
    {
        cudaMalloc(&materials, nbMaterials * sizeof(Material));

        cudaMemcpy(materials, helper.materialsGPU.data(), nbMaterials * sizeof(Material), cudaMemcpyHostToDevice);
    }
    else
    {
        materials = nullptr;
    }
}

void CudaScene::sceneSize(CudaSceneHelper &helper)
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
    totalSize += nbSpheres * sizeof(Sphere);
    printf("Size of spheres: %zu bytes. %2.2f gain compared to v1\n", nbSpheres * sizeof(Sphere),
           (1.f - (nbSpheres * sizeof(Sphere) / 38240.f)) * 100.f);
    totalSize += nbPlanes * sizeof(Plane);
    printf("Size of planes: %zu bytes. %2.2f gain compared to v1\n", nbPlanes * sizeof(Plane),
           (1.f - (nbPlanes * sizeof(Plane) / 20.f)) * 100.f);
    totalSize += nbTriangleMeshes * sizeof(TriangleMesh);
    printf("Size of triangle meshes: %zu bytes. %2.2f gain compared to v1\n", nbTriangleMeshes * sizeof(TriangleMesh),
           (1.f - (nbTriangleMeshes * sizeof(TriangleMesh) / 1.f)) * 100.f);
    totalSize += nbMaterials * sizeof(Material);
    printf("Size of materials: %zu bytes. %2.2f gain compared to v1\n", nbMaterials * sizeof(Material),
           (1.f - (nbMaterials * sizeof(Material) / 15360.f)) * 100.f);
    totalSize += nbLights * sizeof(Light);
    printf("Size of lights: %zu bytes. %2.2f gain compared to v1\n", nbLights * sizeof(Light),
           (1.f - (nbLights * sizeof(Light) / 192.f)) * 100.f);
    totalSize += helper.primitivesGPU.size() * sizeof(BaseObject);
    printf("Size of primitives: %zu bytes. %2.2f gain compared to v1\n",
           helper.primitivesGPU.size() * sizeof(BaseObject),
           (1.f - (helper.primitivesGPU.size() * sizeof(BaseObject) / 22992.f)) * 100.f);
    totalSize += bvhScene.getDeviceSize();
    printf("Size of implicit spheres: %zu bytes. %2.2f gain compared to v1\n",
           nbImplicitSpheres * sizeof(ImplicitSphere),
           (1.f - (nbImplicitSpheres * sizeof(ImplicitSphere) / 400.f)) * 100.f);
    totalSize += nbImplicitSpheres * sizeof(ImplicitSphere);
    printf("BVH size: %zu bytes. %2.2f gain compared to v1\n", bvhScene.getDeviceSize(),
           (1.f - (bvhScene.getDeviceSize() / 68928.f)) * 100.f);
    printf("Total size of GPU data: %zu bytes. %2.2f gain compared to v1\n", totalSize,
           (1.f - (totalSize / 145732.f)) * 100.f);
}

void sortMaterials(std::vector<Material> &materialsGPU, std::vector<int> &planeType, std::vector<int> &sphereType,
                   std::vector<int> &triangleMeshType, std::vector<Material> &lambertList,
                   std::vector<Material> &metalList, std::vector<Material> &plasticList,
                   std::vector<Material> &transparentList, std::vector<Material> &emissiveList,
                   std::vector<Material> &mirrorList, std::vector<Plane> &planesGPU, std::vector<Sphere> &spheresGPU,
                   std::vector<TriangleMesh> &triangleMeshesGPU)
{
    int padding[6];
    padding[0] = 0; // Account for ground plane at index 0
    padding[1] = lambertList.size();
    padding[2] = padding[1] + metalList.size();
    padding[3] = padding[2] + plasticList.size();
    padding[4] = padding[3] + transparentList.size();
    padding[5] = padding[4] + emissiveList.size();

    // adding materials from a same type together to improve memory coherence when shading
    materialsGPU.insert(materialsGPU.end(), lambertList.begin(), lambertList.end());
    materialsGPU.insert(materialsGPU.end(), metalList.begin(), metalList.end());
    materialsGPU.insert(materialsGPU.end(), plasticList.begin(), plasticList.end());
    materialsGPU.insert(materialsGPU.end(), transparentList.begin(), transparentList.end());
    materialsGPU.insert(materialsGPU.end(), emissiveList.begin(), emissiveList.end());
    materialsGPU.insert(materialsGPU.end(), mirrorList.begin(), mirrorList.end());

    for (int i = 0; i < planesGPU.size(); i++)
    {
        Plane p = planesGPU[i];
        planesGPU[i].materialIndex += padding[planeType[i]];
    }

    for (int i = 0; i < spheresGPU.size(); i++)
    {
        Sphere s = spheresGPU[i];
        spheresGPU[i].materialIndex += padding[sphereType[i]];
    }

    for (int i = 0; i < triangleMeshesGPU.size(); i++)
    {
        TriangleMesh tm = triangleMeshesGPU[i];
        triangleMeshesGPU[i].materialIndex += padding[triangleMeshType[i]];
    }
}

CudaScene spheresScene(float4 sunDir)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;

    // ===== PLAN =====
    {
        Plane p;
        p.delta = 0.f;
        p.normal = make_float3(0.f, 1.f, 0.f);

        Material ground = Material::makeMaterial(make_float3(0.5f), LAMBERT, 1.0f);
        helper.lambertList.push_back(ground);
        p.materialIndex = helper.lambertList.size() - 1;
        helper.planeType.push_back(0);
        helper.planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material blueGlass = Material::makeMaterial(make_float3(0.35f, 0.65f, 1.0f), TRANSPARENT, 0.f, 0.f, 1.5f, 0.f);

    Material mirror = Material::makeMaterial(make_float3(1.f, 1.f, 1.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(0.9f, 0.9f, 0.9f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);
    int blueTransparentIdx = helper.transparentList.size();
    helper.transparentList.push_back(blueGlass);
    int mirrorIdx = helper.mirrorList.size();
    helper.mirrorList.push_back(mirror);
    int transparentIdx = helper.transparentList.size();
    helper.transparentList.push_back(transparent);
    int emissiveIdx = helper.emissiveList.size();
    helper.emissiveList.push_back(emissive);

    float bigRadius = 1.0f;
    float smallRadius = 0.2f;
    float margin = 0.05f;
    float minDist = bigRadius + smallRadius + margin;

    int numberOfSpheresPerSide = 10;
    // ===== PETITES SPHERES =====
    for (int i = -numberOfSpheresPerSide; i < numberOfSpheresPerSide; i++)
    {
        for (int j = -numberOfSpheresPerSide; j < numberOfSpheresPerSide; j++)
        {
            double choose_mat = randomDouble();

            float3 center = make_float3(i + 0.9f * randomFloat(), 0.2f, j + 0.9f * randomFloat());

            if (length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(0.f, 1.f, 0.f)) < minDist ||
                length(center - make_float3(-4.f, 1.f, 0.f)) < minDist)
                continue;

            Sphere s;
            s.center1 = center;
            s.center2 = center;
            s.radius = 0.2f;

            // ===== MATERIAL =====
            if (choose_mat < 0.40)
            {
                // Lambert coloré
                Material mat = Material::randomLambert();

                helper.lambertList.push_back(mat);
                s.materialIndex = helper.lambertList.size() - 1;
                helper.sphereType.push_back(0);
            }
            else if (choose_mat < 0.62)
            {
                // Métal coloré
                Material mat = Material::randomMetal();

                helper.metalList.push_back(mat);
                s.materialIndex = helper.metalList.size() - 1;
                helper.sphereType.push_back(1);
            }
            else if (choose_mat < 0.82)
            {
                // Plastique coloré
                Material mat = Material::randomPlastic();

                helper.plasticList.push_back(mat);
                s.materialIndex = helper.plasticList.size() - 1;
                helper.sphereType.push_back(2);
            }
            else if (choose_mat < 0.90)
            {
                // Miroir légèrement bleuté
                s.materialIndex = mirrorIdx;
                helper.sphereType.push_back(5);
            }
            else if (choose_mat < 0.985)
            {
                // Verre coloré aléatoire
                Material mat = Material::randomTransparent();

                helper.transparentList.push_back(mat);
                s.materialIndex = helper.transparentList.size() - 1;
                helper.sphereType.push_back(3);
            }
            else
            {
                // Émissif coloré rare
                Material mat = Material::randomEmissive();

                helper.emissiveList.push_back(mat);
                s.materialIndex = helper.emissiveList.size() - 1;
                helper.sphereType.push_back(4);
                Light light;
                light.metadata = Light::packMetadata(LightType::SPHERE_GEOM, (int)helper.spheresGPU.size());

                helper.lightsGPU.push_back(light);
            }

            // ===== AABB =====
            float3 r = make_float3(s.radius);

            helper.primitivesGPU.push_back(
                BaseObject{center - r, center + r, ObjectType::SPHERE, (int)helper.spheresGPU.size()});
            helper.spheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex, int sphereTypeValue) {
        Sphere s;
        s.center1 = center;
        s.center2 = center;
        s.radius = radius;
        s.materialIndex = matIndex;

        float3 r = make_float3(radius);

        helper.primitivesGPU.push_back(
            BaseObject{center - r, center + r, ObjectType::SPHERE, (int)helper.spheresGPU.size()});

        helper.spheresGPU.push_back(s);
        helper.sphereType.push_back(sphereTypeValue);

        if (sphereTypeValue == 4) // EMISSIVE
        {
            Light l;
            l.metadata = Light::packMetadata(LightType::SPHERE_GEOM, helper.spheresGPU.size() - 1);
            helper.lightsGPU.push_back(l);
        }
    };

    addBigSphere(make_float3(0.f, 1.f, 0.f), 1.f, transparentIdx, 3);
    addBigSphere(make_float3(-4.f, 1.f, 0.f), 1.f, emissiveIdx, 4);
    addBigSphere(make_float3(4.f, 1.f, 0.f), 1.f, mirrorIdx, 5);

    sortMaterials(helper.materialsGPU, helper.planeType, helper.sphereType, helper.triangleMeshType, helper.lambertList,
                  helper.metalList, helper.plasticList, helper.transparentList, helper.emissiveList, helper.mirrorList,
                  helper.planesGPU, helper.spheresGPU, helper.triangleMeshesGPU);

    // ===== LIGHT (SUN) =====
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    helper.lightsGPU.push_back(l);

    // ===== UPLOAD =====
    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);

    gpuScene.sceneSize(helper);
    return gpuScene;
}

CudaScene implicitSpheresScene(float4 sunDir)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;
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

        planesGPU.push_back(p);
    }

    // ===== MATERIALS DE BASE =====
    Material mirror = Material::makeMaterial(make_float3(0.f), MIRROR);
    Material transparent = Material::makeMaterial(make_float3(1.f), TRANSPARENT, 0.f, 0.f, 1.5f);
    Material emissive = Material::makeMaterial(make_float3(1.f, 0.f, 0.f), EMISSIVE, 0.f, 0.f, 1.f, 11.f);

    int mirrorIdx = materialsGPU.size();
    materialsGPU.push_back(mirror);
    int transparentIdx = materialsGPU.size();
    materialsGPU.push_back(transparent);
    int emissiveIdx = materialsGPU.size();
    materialsGPU.push_back(emissive);

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
            double choose_mat = randomDouble();

            float3 center = make_float3(i + 0.9f * randomFloat(), 0.2f, j + 0.9f * randomFloat());

            if (length(center - make_float3(4.f, 1.f, 0.f)) < minDist ||
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
                Material mat =
                    Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), LAMBERT, 1.0f);
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

            primitivesGPU.push_back(
                BaseObject{center - r, center + r, ObjectType::IMPLICIT_SPHERE, (int)implicitSpheresGPU.size()});
            implicitSpheresGPU.push_back(s);
        }
    }

    // ===== GROSSES SPHERES =====
    auto addBigSphere = [&](float3 center, float radius, int matIndex) {
        ImplicitSphere s;
        s.center1 = center;
        s.center2 = center;
        s.radius = radius;
        s.materialIndex = matIndex;

        float3 r = make_float3(radius);

        primitivesGPU.push_back(
            BaseObject{center - r, center + r, ObjectType::IMPLICIT_SPHERE, (int)implicitSpheresGPU.size()});

        implicitSpheresGPU.push_back(s);
        Material &mat = materialsGPU[matIndex];

        if (mat.type() == MaterialType::EMISSIVE)
        {
            Light l;
            l.metadata = Light::packMetadata(LightType::IMPLICIT_SPHERE_GEOM, implicitSpheresGPU.size() - 1);
            lightsGPU.push_back(l);
        }
    };
    /*
    MeshAndPrimitive meshAndPrim = loadTriangleMesh("../data/bunny/Bunny.obj", materialsGPU.size() - 1,
    triangleMeshesGPU.size()); triangleMeshesGPU.push_back(meshAndPrim.mesh); primitivesGPU.push_back(meshAndPrim.prim);
    */

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
    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);

    gpuScene.sceneSize(helper);
    return gpuScene;
}

CudaScene singleObject(float4 sunDir)
{
    CudaScene gpuScene;
    CudaSceneHelper helper;

    std::vector<TriangleMesh> triangleMeshesGPU;
    std::vector<BaseObject> primitivesGPU;
    std::vector<Material> materialsGPU;
    std::vector<Light> lightsGPU;
    Material mat = Material::makeMaterial(make_float3(randomFloat(), randomFloat(), randomFloat()), LAMBERT, 1.0f);
    materialsGPU.push_back(mat);
    Quaternion rotation = quaternionFromAxisAngle(make_float3(0.f, 1.f, 0.f), 00.f);
    MeshAndPrimitive meshAndPrim =
        loadTriangleMesh("../data/bunny/Bunny.obj", materialsGPU.size() - 1, triangleMeshesGPU.size(),
                         make_float3(2.f, 2.f, 2.f), rotation, make_float3(0.f, 0.f, 0.f));
    triangleMeshesGPU.push_back(meshAndPrim.mesh);
    primitivesGPU.push_back(meshAndPrim.prim);
    Light l;
    l.color_power = make_float4(1.f, 0.95f, 0.9f, 100.f);
    l.direction = make_float4(sunDir.x, sunDir.y, sunDir.z, 0.f);
    l.metadata = Light::packMetadata(LightType::SUN, 0);
    lightsGPU.push_back(l);
    gpuScene.uploadObjects(helper);
    gpuScene.uploadLights(helper);
    gpuScene.uploadMaterials(helper);
    gpuScene.sceneSize(helper);
    return gpuScene;
}
