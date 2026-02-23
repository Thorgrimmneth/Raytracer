#include "cuda_bvh_scene.cuh"
#include <cstdio>
#include <algorithm>

__host__
BVHScene BVHScene::buildBVHScene(std::vector<Sphere>* objects)
{
    BVHScene scene{};

    if (!objects || objects->empty())
        return scene;

    const int maxObjectsPerLeaf = 8;
    const int maxDepth = 32;
    const int BIN_COUNT = 16;

    std::vector<BVHSceneNode> nodes;
    nodes.reserve(objects->size() * 2);

    // Root
    nodes.push_back(BVHSceneNode{});

    std::vector<BuildTask> stack;
    stack.push_back({0, 0, (int)objects->size(), 0});

    while (!stack.empty())
    {
        BuildTask task = stack.back();
        stack.pop_back();

        BVHSceneNode& node = nodes[task.nodeIndex];

        node.firstObjectIndex = task.first;
        node.lastObjectIndex  = task.last;

        AABB bbox{};
        for (int i = task.first; i < task.last; ++i)
            bbox.extend((*objects)[i].base.bbox);

        node.bbox = bbox;

        int nbObjects = task.last - task.first;

        if (nbObjects <= maxObjectsPerLeaf || task.depth >= maxDepth)
            continue;

        AABB centroidBBox{};
        for (int i = task.first; i < task.last; ++i)
            centroidBBox.extend((*objects)[i].base.bbox.centroid());

        float bestCost = std::numeric_limits<float>::max();
        int bestAxis = -1;
        int bestSplitBin = -1;

        for (int axis = 0; axis < 3; axis++)
        {
            float cmin = getAxis(centroidBBox.min,axis);
            float cmax = getAxis(centroidBBox.max,axis);
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
                float centroid = getAxis((*objects)[i].base.bbox.centroid(),axis);

                int binId = int(BIN_COUNT * (centroid - cmin) / extent);
                binId = std::min(BIN_COUNT - 1, std::max(0, binId));

                bins[binId].count++;
                bins[binId].bbox.extend((*objects)[i].base.bbox);
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

            float parentArea = node.bbox.area();

            // Evaluate splits
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
            continue;

        float cmin = getAxis(centroidBBox.min,bestAxis);
        float cmax = getAxis(centroidBBox.max,bestAxis);
        float extent = cmax - cmin;

        float splitPos =
            cmin + extent * float(bestSplitBin + 1) / float(BIN_COUNT);

        auto midIter = std::partition(
            objects->begin() + task.first,
            objects->begin() + task.last,
            [bestAxis, splitPos](const Sphere& s)
            {
                return getAxis(s.base.bbox.centroid(),bestAxis) < splitPos;
            }
        );

        int mid = midIter - objects->begin();

        if (mid == task.first || mid == task.last)
            continue;

        int leftIndex = nodes.size();
        nodes.push_back(BVHSceneNode{});

        int rightIndex = nodes.size();
        nodes.push_back(BVHSceneNode{});

        node.left  = leftIndex;
        node.right = rightIndex;

        stack.push_back({rightIndex, mid, task.last, task.depth + 1});
        stack.push_back({leftIndex,  task.first, mid, task.depth + 1});
    }

    cudaMalloc(&scene.d_objects,
               objects->size() * sizeof(Sphere));

    cudaMemcpy(scene.d_objects,
               objects->data(),
               objects->size() * sizeof(Sphere),
               cudaMemcpyHostToDevice);

    cudaMalloc(&scene.d_nodes,
               nodes.size() * sizeof(BVHSceneNode));

    cudaMemcpy(scene.d_nodes,
               nodes.data(),
               nodes.size() * sizeof(BVHSceneNode),
               cudaMemcpyHostToDevice);

    scene.nbObjects = objects->size();
    scene.nbNodes   = nodes.size();

    return scene;
}

__device__ 
bool BVHScene::intersect(const Ray &p_ray,
                         const float p_tMin,
                         const float p_tMax,
                         HitRecord &p_hitRecord) const
{
    int stack[64];
    int stackPtr = 0;
    bool hit = false;

    float tMax = p_tMax;

    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        int nodeIndex = stack[--stackPtr];
        const BVHSceneNode& node = d_nodes[nodeIndex];

        if (!node.bbox.intersect(p_ray, p_tMin, tMax))
            continue;

        if (node.isLeaf())
        {
            for (int i = node.firstObjectIndex;
                 i < node.lastObjectIndex;
                 ++i)
            {
                if (d_objects[i].intersect(p_ray, p_tMin, tMax, p_hitRecord))
                {
                    tMax = p_hitRecord.distance;
                    hit = true;
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return hit;
}

__device__ 
bool BVHScene::intersectAny(const Ray &p_ray, const float p_tMin, const float p_tMax) const{
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // root index

    while(stackPtr > 0)
    {
        int nodeIndex = stack[--stackPtr];
        const BVHSceneNode& node = d_nodes[nodeIndex];

        if (!node.bbox.intersect(p_ray, p_tMin, p_tMax))
            continue;

        if (node.isLeaf())
        {
            for(int i = node.firstObjectIndex;
                i < node.lastObjectIndex;
                ++i)
            {
                if(d_objects[i].intersectAny(p_ray, p_tMin, p_tMax)) return true;
            }
        }
        else
        {
            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.right;
        }
    }

    return false;
}