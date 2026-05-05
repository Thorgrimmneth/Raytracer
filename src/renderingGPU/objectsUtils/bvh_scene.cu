#include "bvh_scene.cuh"
#include <algorithm>
#include <limits>

// ============================================================
// BUILD BVH (CPU)
// ============================================================

__host__
BVHScene BVHScene::buildBVHScene(std::vector<BaseObject>* primitives,
                                 std::vector<Sphere>* spheres,
                                 std::vector<Plane>* planes,
                                 std::vector<TriangleMesh>* meshes)
{
    BVHScene scene{};

    if (!primitives || primitives->empty())
        return scene;

    const int maxObjectsPerLeaf = 1;
    const int maxDepth = 16;
    const int BIN_COUNT = 16;

    std::vector<BVHSceneNode> nodes;
    nodes.reserve(primitives->size() * 2);

    // ✅ indices (IMPORTANT)
    std::vector<int> indices(primitives->size());
    for (int i = 0; i < primitives->size(); i++)
        indices[i] = i;

    // Root
    nodes.push_back(BVHSceneNode{});

    std::vector<BuildTask> stack;
    stack.push_back({0, 0, (int)primitives->size(), 0});

    while (!stack.empty())
    {
        BuildTask task = stack.back();
        stack.pop_back();

        int nodeIndex = task.nodeIndex;

        nodes[nodeIndex].firstObjectIndex = task.first;
        nodes[nodeIndex].lastObjectIndex  = task.last;

        // ===== Compute bbox =====
        AABB bbox{};
        for (int i = task.first; i < task.last; ++i)
            bbox.extend((*primitives)[indices[i]].bbox);

        nodes[nodeIndex].bbox = bbox;

        int nbObjects = task.last - task.first;

        if (nbObjects <= maxObjectsPerLeaf || task.depth >= maxDepth)
            continue;

        // ===== Centroid bbox =====
        AABB centroidBBox{};
        for (int i = task.first; i < task.last; ++i)
            centroidBBox.extend((*primitives)[indices[i]].bbox.centroid());

        float bestCost = std::numeric_limits<float>::max();
        int bestAxis = -1;
        int bestSplitBin = -1;

        for (int axis = 0; axis < 3; axis++)
        {
            float cmin = getAxis(centroidBBox.min, axis);
            float cmax = getAxis(centroidBBox.max, axis);
            float extent = cmax - cmin;

            if (extent <= 1e-5f)
                continue;

            struct Bin {
                AABB bbox;
                int count = 0;
            };

            Bin bins[BIN_COUNT];

            // Fill bins
            for (int i = task.first; i < task.last; i++)
            {
                float centroid =
                    getAxis((*primitives)[indices[i]].bbox.centroid(), axis);

                int binId = int(BIN_COUNT * (centroid - cmin) / extent);
                binId = std::min(BIN_COUNT - 1, std::max(0, binId));

                bins[binId].count++;
                bins[binId].bbox.extend((*primitives)[indices[i]].bbox);
            }

            // Prefix
            AABB leftBBox[BIN_COUNT];
            int leftCount[BIN_COUNT];

            AABB tmpBox{};
            int tmpCount = 0;

            for (int i = 0; i < BIN_COUNT; i++)
            {
                tmpBox.extend(bins[i].bbox);
                tmpCount += bins[i].count;

                leftBBox[i] = tmpBox;
                leftCount[i] = tmpCount;
            }

            // Suffix
            AABB rightBBox[BIN_COUNT];
            int rightCount[BIN_COUNT];

            tmpBox = AABB{};
            tmpCount = 0;

            for (int i = BIN_COUNT - 1; i >= 0; i--)
            {
                tmpBox.extend(bins[i].bbox);
                tmpCount += bins[i].count;

                rightBBox[i] = tmpBox;
                rightCount[i] = tmpCount;
            }

            for (int i = 0; i < BIN_COUNT - 1; i++)
            {
                if (leftCount[i] == 0 || rightCount[i + 1] == 0)
                    continue;

                float cost =
                    leftBBox[i].area() * leftCount[i] +
                    rightBBox[i + 1].area() * rightCount[i + 1];

                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplitBin = i;
                }
            }
        }

        // ===== fallback =====
        if (bestAxis == -1)
        {
            int mid = task.first + nbObjects / 2;

            int leftIndex = nodes.size();
            nodes.push_back(BVHSceneNode{});

            int rightIndex = nodes.size();
            nodes.push_back(BVHSceneNode{});

            nodes[nodeIndex].left = leftIndex;
            nodes[nodeIndex].right = rightIndex;

            stack.push_back({rightIndex, mid, task.last, task.depth + 1});
            stack.push_back({leftIndex, task.first, mid, task.depth + 1});
            continue;
        }

        float cmin = getAxis(centroidBBox.min, bestAxis);
        float extent = getAxis(centroidBBox.max, bestAxis) - cmin;

        float splitPos =
            cmin + extent * float(bestSplitBin + 1) / float(BIN_COUNT);

        auto midIter = std::partition(
            indices.begin() + task.first,
            indices.begin() + task.last,
            [&](int idx)
            {
                return getAxis((*primitives)[idx].bbox.centroid(), bestAxis) < splitPos;
            }
        );

        int mid = midIter - indices.begin();

        if (mid == task.first || mid == task.last)
            continue;

        int leftIndex = nodes.size();
        nodes.push_back(BVHSceneNode{});

        int rightIndex = nodes.size();
        nodes.push_back(BVHSceneNode{});

        nodes[nodeIndex].left  = leftIndex;
        nodes[nodeIndex].right = rightIndex;

        stack.push_back({rightIndex, mid, task.last, task.depth + 1});
        stack.push_back({leftIndex,  task.first, mid, task.depth + 1});
    }

    // ===== Upload indices =====
    cudaMalloc(&scene.d_indices,
               indices.size() * sizeof(int));

    cudaMemcpy(scene.d_indices,
               indices.data(),
               indices.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // ===== Upload nodes =====
    cudaMalloc(&scene.d_nodes,
               nodes.size() * sizeof(BVHSceneNode));

    cudaMemcpy(scene.d_nodes,
               nodes.data(),
               nodes.size() * sizeof(BVHSceneNode),
               cudaMemcpyHostToDevice);

    scene.d_primitives = nullptr;
    scene.nbObjects = primitives->size();
    scene.nbNodes   = nodes.size();

    return scene;
}

// ============================================================
// INTERSECT (closest hit)
// ============================================================

__device__ __noinline__
bool BVHScene::intersect(const Ray &ray,
                         const float tMin,
                         const float tMaxInit,
                         HitRecord &hit) const
{
    Current stack[64];
    int stackPtr = 0;

    float tMax = tMaxInit;
    bool hitSomething = false;

    float dist;
    if (!d_nodes[0].bbox.intersectCheck(ray, tMin, tMax, dist))
        return false;

    stack[stackPtr++] = {0, dist};

    while (stackPtr > 0)
    {
        Current current = stack[--stackPtr];

        if (current.distance > tMax)
            continue;

        const BVHSceneNode& node = d_nodes[current.index];

        if (node.isLeaf())
        {
            for (int i = node.firstObjectIndex; i < node.lastObjectIndex; ++i)
            {
                BaseObject& prim = d_primitives[d_indices[i]];

                switch (prim.type)
                {
                    case SPHERE:
                        if (d_spheres[prim.index].intersect(ray, tMin, tMax, hit))
                        {
                            tMax = hit.distance;
                            hitSomething = true;
                        }
                        break;

                    case PLANE:
                        if (d_planes[prim.index].intersect(ray, tMin, tMax, hit))
                        {
                            tMax = hit.distance;
                            hitSomething = true;
                        }
                        break;

                    case TRIANGLE:
                        if (d_meshes[prim.index].intersect(ray, tMin, tMax, hit))
                        {
                            tMax = hit.distance;
                            hitSomething = true;
                        }
                        break;
                }
            }
        }
        else
        {
            float dl, dr;
            bool hl = d_nodes[node.left].bbox.intersectCheck(ray, tMin, tMax, dl);
            bool hr = d_nodes[node.right].bbox.intersectCheck(ray, tMin, tMax, dr);

            if (hl && hr)
            {
                if (dl < dr)
                {
                    stack[stackPtr++] = {node.right, dr};
                    stack[stackPtr++] = {node.left, dl};
                }
                else
                {
                    stack[stackPtr++] = {node.left, dl};
                    stack[stackPtr++] = {node.right, dr};
                }
            }
            else if (hl)
                stack[stackPtr++] = {node.left, dl};
            else if (hr)
                stack[stackPtr++] = {node.right, dr};
        }
    }

    return hitSomething;
}

size_t BVHScene::getDeviceSize() const
{
    size_t size = 0;
    size += nbNodes * sizeof(BVHSceneNode);
    size += nbObjects * sizeof(int); // d_indices
    return size;
}