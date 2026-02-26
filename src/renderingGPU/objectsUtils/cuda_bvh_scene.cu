#include "cuda_bvh_scene.cuh"
#include <cstdio>
#include <algorithm>

__host__
BVHScene BVHScene::buildBVHScene(std::vector<BaseObject>* primitives,
                                 std::vector<Sphere>* spheres,
                                 std::vector<Plane>* planes,
                                 std::vector<TriangleMesh>* meshes)
{
    BVHScene scene{};

    if (!primitives || primitives->empty())
        return scene;

    const int maxObjectsPerLeaf = 8;
    const int maxDepth = 32;
    const int BIN_COUNT = 16;

    std::vector<BVHSceneNode> nodes;
    nodes.reserve(primitives->size() * 2);

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

        // ==== Compute bbox ====
        AABB bbox{};
        for (int i = task.first; i < task.last; ++i)
            bbox.extend((*primitives)[i].bbox);

        nodes[nodeIndex].bbox = bbox;

        int nbObjects = task.last - task.first;

        if (nbObjects <= maxObjectsPerLeaf || task.depth >= maxDepth)
            continue;

        // ==== Centroid bbox ====
        AABB centroidBBox{};
        for (int i = task.first; i < task.last; ++i)
            centroidBBox.extend((*primitives)[i].bbox.centroid());

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
                    getAxis((*primitives)[i].bbox.centroid(), axis);

                int binId =
                    int(BIN_COUNT * (centroid - cmin) / extent);

                binId = std::min(BIN_COUNT - 1,
                                 std::max(0, binId));

                bins[binId].count++;
                bins[binId].bbox.extend((*primitives)[i].bbox);
            }

            // Prefix sweep
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

            // Suffix sweep
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

        if (bestAxis == -1)
        {
            // fallback split médian
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
        float cmax = getAxis(centroidBBox.max, bestAxis);
        float extent = cmax - cmin;

        float splitPos =
            cmin + extent * float(bestSplitBin + 1) / float(BIN_COUNT);

        auto midIter = std::partition(
            primitives->begin() + task.first,
            primitives->begin() + task.last,
            [bestAxis, splitPos](const BaseObject& obj)
            {
                return getAxis(obj.bbox.centroid(), bestAxis) < splitPos;
            }
        );

        int mid = midIter - primitives->begin();

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

    // ==== Upload primitives ====
    cudaMalloc(&scene.d_primitives,
               primitives->size() * sizeof(BaseObject));

    cudaMemcpy(scene.d_primitives,
               primitives->data(),
               primitives->size() * sizeof(BaseObject),
               cudaMemcpyHostToDevice);

    // ==== Upload nodes ====
    cudaMalloc(&scene.d_nodes,
               nodes.size() * sizeof(BVHSceneNode));

    cudaMemcpy(scene.d_nodes,
               nodes.data(),
               nodes.size() * sizeof(BVHSceneNode),
               cudaMemcpyHostToDevice);

    scene.nbObjects = primitives->size();
    scene.nbNodes   = nodes.size();

    return scene;
}

__device__ __noinline__
bool BVHScene::intersect(const Ray &p_ray,
                         const float p_tMin,
                         const float p_tMax,
                         HitRecord &p_hitRecord) const
{
    Current stack[256];
    int stackPtr = 0;
    bool hit = false;

    float tMax = p_tMax;

    float distTemp;
    if(!d_nodes[0].bbox.intersectCheck(p_ray, p_tMin, p_tMax, distTemp)) return false;
    if(distTemp > tMax) return false;
    stack[stackPtr++] = {0,distTemp};

    while (stackPtr > 0)
    {
        Current currentNode = stack[--stackPtr];
        if(currentNode.distance > tMax) continue;
        const BVHSceneNode& node = d_nodes[currentNode.index];

        if (node.isLeaf())
        {
            for (int i = node.firstObjectIndex;
                 i < node.lastObjectIndex;
                 ++i)
            {
                BaseObject& prim = d_primitives[i];
                switch(prim.type)
                {
                    case ObjectType::SPHERE:
                        if (d_spheres[prim.index].intersect(p_ray, p_tMin, tMax, p_hitRecord))
                        {
                            tMax = p_hitRecord.distance;
                            hit = true;
                        }
                        break;
                    case ObjectType::PLANE:
                        if(d_planes[prim.index].intersect(p_ray, p_tMin, tMax, p_hitRecord))
                        {
                            tMax = p_hitRecord.distance;
                            hit = true;
                        }
                        break;
                    case ObjectType::TRIANGLE:
                        if(d_meshes[prim.index].intersect(p_ray, p_tMin, tMax, p_hitRecord))
                        {
                            tMax = p_hitRecord.distance;
                            hit = true;
                        }
                        break;
                }
                
            }
        }
        else
        {
            float tempLeft;
            bool hitLeft;
            hitLeft = d_nodes[node.left].bbox.intersectCheck(p_ray, p_tMin, tMax, tempLeft);

            float tempRight;
            bool hitRight;
            hitRight = d_nodes[node.right].bbox.intersectCheck(p_ray, p_tMin, tMax, tempRight);

            if(!hitLeft && !hitRight) continue;
            if(hitLeft && hitRight){
                if(tempLeft < tempRight){
                    stack[stackPtr++] = Current{node.right, tempRight};
                    stack[stackPtr++] = Current{node.left, tempLeft};
                }
                else{
                    stack[stackPtr++] = Current{node.left, tempLeft};
                    stack[stackPtr++] = Current{node.right, tempRight};
                }
            }
            else{
                if(hitLeft)stack[stackPtr++] = Current{node.left, tempLeft};
                else stack[stackPtr++] = Current{node.right, tempRight};
            }
        }
    }

    return hit;
}

