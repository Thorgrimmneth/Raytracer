
#include "sbvh.cuh"

SplitCandidate findBestSpatialSplit(
    const std::vector<TriangleRef>& refs,
    int start,
    int end,
    const AABB& parentBox,
    const BVHBuildConfig& config)
{
    SplitCandidate best;
    best.type = SPLIT_SPATIAL;

    float parentArea = parentBox.area();
    if (parentArea <= 0.f)
        return best;

    for (int axis = 0; axis < 3; ++axis) {
        float minA = getAxisMin(parentBox, axis);
        float maxA = getAxisMax(parentBox, axis);
        float extent = maxA - minA;

        if (extent <= 1e-8f)
            continue;

        for (int b = 1; b < config.binCount; ++b) {
            float pos = minA + extent * ((float)b / (float)config.binCount);

            AABB leftBox;
            AABB rightBox;
            int leftCount = 0;
            int rightCount = 0;
            int duplicates = 0;

            for (int i = start; i < end; ++i) {
                const TriangleRef& ref = refs[i];

                float bmin = getAxisMin(ref.bbox, axis);
                float bmax = getAxisMax(ref.bbox, axis);

                if (bmax <= pos) {
                    leftBox.extend(ref.bbox);
                    leftCount++;
                }
                else if (bmin >= pos) {
                    rightBox.extend(ref.bbox);
                    rightCount++;
                }
                else {
                    AABB clippedLeft = ref.bbox;
                    AABB clippedRight = ref.bbox;

                    setAxisMax(clippedLeft, axis, pos);
                    setAxisMin(clippedRight, axis, pos);

                    if (clippedLeft.isValid()) {
                        leftBox.extend(clippedLeft);
                        leftCount++;
                    }

                    if (clippedRight.isValid()) {
                        rightBox.extend(clippedRight);
                        rightCount++;
                    }

                    duplicates++;
                }
            }

            if (leftCount == 0 || rightCount == 0)
                continue;

            float leftArea = leftBox.area();
            float rightArea = rightBox.area();

            float cost =
                config.traversalCost +
                config.intersectionCost *
                ((leftArea / parentArea) * leftCount +
                 (rightArea / parentArea) * rightCount);

            // Petite pénalité pour éviter trop de duplication
            cost += 0.05f * duplicates;

            if (cost < best.cost) {
                best.cost = cost;
                best.axis = axis;
                best.pos = pos;
            }
        }
    }

    return best;
}

SplitCandidate findBestObjectSplit(
    std::vector<TriangleRef>& refs,
    int start,
    int end,
    const AABB& parentBox,
    const BVHBuildConfig& config)
{
    SplitCandidate best;
    best.type = SPLIT_OBJECT;

    int count = end - start;
    if (count <= config.maxLeafSize)
        return best;

    float parentArea = parentBox.area();
    if (parentArea <= 0.f)
        return best;

    for (int axis = 0; axis < 3; ++axis) {
        std::sort(
            refs.begin() + start,
            refs.begin() + end,
            [axis](const TriangleRef& a, const TriangleRef& b) {
                return getAxis(a.centroid, axis) < getAxis(b.centroid, axis);
            }
        );

        std::vector<AABB> leftBoxes(count);
        std::vector<AABB> rightBoxes(count);

        AABB leftAccum;
        for (int i = 0; i < count; ++i) {
            leftAccum.extend(refs[start + i].bbox);
            leftBoxes[i] = leftAccum;
        }

        AABB rightAccum;
        for (int i = count - 1; i >= 0; --i) {
            rightAccum.extend(refs[start + i].bbox);
            rightBoxes[i] = rightAccum;
        }

        for (int i = 1; i < count; ++i) {
            int leftCount = i;
            int rightCount = count - i;

            float leftArea = leftBoxes[i - 1].area();
            float rightArea = rightBoxes[i].area();

            float cost =
                config.traversalCost +
                config.intersectionCost *
                ((leftArea / parentArea) * leftCount +
                 (rightArea / parentArea) * rightCount);

            if (cost < best.cost) {
                best.cost = cost;
                best.axis = axis;

                float c0 = getAxis(refs[start + i - 1].centroid, axis);
                float c1 = getAxis(refs[start + i].centroid, axis);
                best.pos = 0.5f * (c0 + c1);
            }
        }
    }

    return best;
}

int applyObjectSplit(
    std::vector<TriangleRef>& refs,
    int start,
    int end,
    const SplitCandidate& split)
{
    int mid = start;

    for (int i = start; i < end; ++i) {
        float c = getAxis(refs[i].centroid, split.axis);

        if (c < split.pos) {
            std::swap(refs[i], refs[mid]);
            mid++;
        }
    }

    if (mid == start || mid == end) {
        mid = start + (end - start) / 2;

        std::nth_element(
            refs.begin() + start,
            refs.begin() + mid,
            refs.begin() + end,
            [&split](const TriangleRef& a, const TriangleRef& b) {
                return getAxis(a.centroid, split.axis) <
                       getAxis(b.centroid, split.axis);
            }
        );
    }

    return mid;
}

int applySpatialSplit(
    std::vector<TriangleRef>& refs,
    int start,
    int end,
    const SplitCandidate& split)
{
    std::vector<TriangleRef> leftRefs;
    std::vector<TriangleRef> rightRefs;

    leftRefs.reserve(end - start);
    rightRefs.reserve(end - start);

    for (int i = start; i < end; ++i) {
        TriangleRef ref = refs[i];

        float bmin = getAxisMin(ref.bbox, split.axis);
        float bmax = getAxisMax(ref.bbox, split.axis);

        if (bmax <= split.pos) {
            leftRefs.push_back(ref);
        }
        else if (bmin >= split.pos) {
            rightRefs.push_back(ref);
        }
        else {
            TriangleRef leftRef = ref;
            TriangleRef rightRef = ref;

            setAxisMax(leftRef.bbox, split.axis, split.pos);
            setAxisMin(rightRef.bbox, split.axis, split.pos);

            leftRef.centroid = make_float3(
                0.5f * (leftRef.bbox.min.x + leftRef.bbox.max.x),
                0.5f * (leftRef.bbox.min.y + leftRef.bbox.max.y),
                0.5f * (leftRef.bbox.min.z + leftRef.bbox.max.z)
            );

            rightRef.centroid = make_float3(
                0.5f * (rightRef.bbox.min.x + rightRef.bbox.max.x),
                0.5f * (rightRef.bbox.min.y + rightRef.bbox.max.y),
                0.5f * (rightRef.bbox.min.z + rightRef.bbox.max.z)
            );

            if (leftRef.bbox.isValid())
                leftRefs.push_back(leftRef);

            if (rightRef.bbox.isValid())
                rightRefs.push_back(rightRef);
        }
    }

    int leftCount = (int)leftRefs.size();
    int rightCount = (int)rightRefs.size();

    if (leftCount == 0 || rightCount == 0)
        return -1;

    std::vector<TriangleRef> merged;
    merged.reserve(leftCount + rightCount);

    merged.insert(merged.end(), leftRefs.begin(), leftRefs.end());
    merged.insert(merged.end(), rightRefs.begin(), rightRefs.end());

    refs.erase(refs.begin() + start, refs.begin() + end);
    refs.insert(refs.begin() + start, merged.begin(), merged.end());

    return start + leftCount;
}

int buildSBVHRecursive(
    std::vector<BVH>& nodes,
    std::vector<int>& finalRefs,
    const std::vector<TriangleRef>& refs,
    const BVHBuildConfig& config,
    int depth)
{
    int nodeIdx = (int)nodes.size();
    nodes.push_back(BVH());

    BVH& node = nodes[nodeIdx];

    AABB nodeBox;
    for (const TriangleRef& ref : refs)
        nodeBox.extend(ref.bbox);

    node.bbox = nodeBox;

    int count = (int)refs.size();

    if (count <= config.maxLeafSize || depth >= config.maxDepth) {
        node.left = -1;
        node.right = -1;
        node.firstRefIndex = (int)finalRefs.size();
        node.refCount = count;

        for (const TriangleRef& ref : refs)
            finalRefs.push_back(ref.triIndex);

        return nodeIdx;
    }

    float leafCost = config.intersectionCost * count;

    std::vector<TriangleRef> mutableRefs = refs;

    SplitCandidate objectSplit =
        findBestObjectSplit(mutableRefs, 0, count, nodeBox, config);

    SplitCandidate bestSplit = objectSplit;

    if (config.enableSpatialSplits) {
        SplitCandidate spatialSplit =
            findBestSpatialSplit(refs, 0, count, nodeBox, config);

        if (spatialSplit.cost < bestSplit.cost) {
            bestSplit = spatialSplit;
        }
    }

    if (bestSplit.type == SPLIT_NONE || bestSplit.cost >= leafCost) {
        node.left = -1;
        node.right = -1;
        node.firstRefIndex = (int)finalRefs.size();
        node.refCount = count;

        for (const TriangleRef& ref : refs)
            finalRefs.push_back(ref.triIndex);

        return nodeIdx;
    }

    std::vector<TriangleRef> leftRefs;
    std::vector<TriangleRef> rightRefs;

    if (bestSplit.type == SPLIT_OBJECT) {
        leftRefs.reserve(count);
        rightRefs.reserve(count);

        for (const TriangleRef& ref : refs) {
            float c = getAxis(ref.centroid, bestSplit.axis);

            if (c < bestSplit.pos)
                leftRefs.push_back(ref);
            else
                rightRefs.push_back(ref);
        }

        if (leftRefs.empty() || rightRefs.empty()) {
            std::vector<TriangleRef> sorted = refs;

            std::nth_element(
                sorted.begin(),
                sorted.begin() + count / 2,
                sorted.end(),
                [&bestSplit](const TriangleRef& a, const TriangleRef& b) {
                    return getAxis(a.centroid, bestSplit.axis) <
                           getAxis(b.centroid, bestSplit.axis);
                }
            );

            leftRefs.assign(sorted.begin(), sorted.begin() + count / 2);
            rightRefs.assign(sorted.begin() + count / 2, sorted.end());
        }
    }
    else {
        leftRefs.reserve(count);
        rightRefs.reserve(count);

        for (const TriangleRef& ref : refs) {
            float bmin = getAxisMin(ref.bbox, bestSplit.axis);
            float bmax = getAxisMax(ref.bbox, bestSplit.axis);

            if (bmax <= bestSplit.pos) {
                leftRefs.push_back(ref);
            }
            else if (bmin >= bestSplit.pos) {
                rightRefs.push_back(ref);
            }
            else {
                TriangleRef leftRef = ref;
                TriangleRef rightRef = ref;

                setAxisMax(leftRef.bbox, bestSplit.axis, bestSplit.pos);
                setAxisMin(rightRef.bbox, bestSplit.axis, bestSplit.pos);

                leftRef.centroid = make_float3(
                    0.5f * (leftRef.bbox.min.x + leftRef.bbox.max.x),
                    0.5f * (leftRef.bbox.min.y + leftRef.bbox.max.y),
                    0.5f * (leftRef.bbox.min.z + leftRef.bbox.max.z)
                );

                rightRef.centroid = make_float3(
                    0.5f * (rightRef.bbox.min.x + rightRef.bbox.max.x),
                    0.5f * (rightRef.bbox.min.y + rightRef.bbox.max.y),
                    0.5f * (rightRef.bbox.min.z + rightRef.bbox.max.z)
                );

                if (leftRef.bbox.isValid())
                    leftRefs.push_back(leftRef);

                if (rightRef.bbox.isValid())
                    rightRefs.push_back(rightRef);
            }
        }
    }

    if (leftRefs.empty() || rightRefs.empty()) {
        node.left = -1;
        node.right = -1;
        node.firstRefIndex = (int)finalRefs.size();
        node.refCount = count;

        for (const TriangleRef& ref : refs)
            finalRefs.push_back(ref.triIndex);

        return nodeIdx;
    }

    int leftChild = buildSBVHRecursive(
        nodes,
        finalRefs,
        leftRefs,
        config,
        depth + 1
    );

    int rightChild = buildSBVHRecursive(
        nodes,
        finalRefs,
        rightRefs,
        config,
        depth + 1
    );

    nodes[nodeIdx].left = leftChild;
    nodes[nodeIdx].right = rightChild;
    nodes[nodeIdx].firstRefIndex = -1;
    nodes[nodeIdx].refCount = 0;

    return nodeIdx;
}

__host__
BVH* buildSBVH(
    TriangleMeshGeometry* triangles,
    int triangleCount,
    float3* vertices,
    float3* normals,
    float2* uvs,
    int& outNodeCount,
    int*& outTriangleRefIndices,
    int& outRefCount)
{
    BVHBuildConfig config;

    std::vector<TriangleRef> initialRefs;
    initialRefs.reserve(triangleCount);

    for (int i = 0; i < triangleCount; ++i) {
        const TriangleMeshGeometry& tri = triangles[i];

        AABB box;
        box.extend(vertices[tri.i0]);
        box.extend(vertices[tri.i1]);
        box.extend(vertices[tri.i2]);

        float3 c = make_float3(
            (vertices[tri.i0].x + vertices[tri.i1].x + vertices[tri.i2].x) / 3.f,
            (vertices[tri.i0].y + vertices[tri.i1].y + vertices[tri.i2].y) / 3.f,
            (vertices[tri.i0].z + vertices[tri.i1].z + vertices[tri.i2].z) / 3.f
        );

        initialRefs.push_back({ i, box, c });
    }

    std::vector<BVH> nodes;
    nodes.reserve(triangleCount * 2);

    std::vector<int> finalRefs;
    finalRefs.reserve((int)(triangleCount * config.maxDuplicationRatio));

    buildSBVHRecursive(
        nodes,
        finalRefs,
        initialRefs,
        config,
        0
    );

    BVH* d_bvhNodes = nullptr;
    cudaMalloc(&d_bvhNodes, nodes.size() * sizeof(BVH));
    cudaMemcpy(
        d_bvhNodes,
        nodes.data(),
        nodes.size() * sizeof(BVH),
        cudaMemcpyHostToDevice
    );

    int* d_triangleRefIndices = nullptr;
    cudaMalloc(&d_triangleRefIndices, finalRefs.size() * sizeof(int));
    cudaMemcpy(
        d_triangleRefIndices,
        finalRefs.data(),
        finalRefs.size() * sizeof(int),
        cudaMemcpyHostToDevice
    );

    outNodeCount = (int)nodes.size();
    outRefCount = (int)finalRefs.size();
    outTriangleRefIndices = d_triangleRefIndices;

    return d_bvhNodes;
}