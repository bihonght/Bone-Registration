#ifndef SDF_VOXEL_H_
#define SDF_VOXEL_H_

#include <vector>
#include <Eigen/Dense>
#include "hash_map.h"

struct SdfVoxel {
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    float dist;
    Eigen::Vector3f grad;
    float weight;
    int closest_face;

    SdfVoxel() :
        dist(0.),
        grad(Eigen::Vector3f::Zero()),
        weight(0.)
    {}
};

using Vec8f = Eigen::Matrix<float, 8, 1>;

struct SdfVoxelHr {

EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec8f d = Vec8f::Zero();
    Vec8f r = Vec8f::Zero();
    Vec8f g = Vec8f::Zero();
    Vec8f b = Vec8f::Zero();

    const float dist;
    const float weight;
    const Eigen::Vector3f grad;

    SdfVoxelHr() :
        dist(0.),
        grad(Eigen::Vector3f::Zero()),
        weight(0.),
        d(Vec8f::Zero()),
        r(Vec8f::Zero()),
        g(Vec8f::Zero()),
        b(Vec8f::Zero())
    {}

    SdfVoxelHr(const SdfVoxel &voxel, const float voxel_size) :
        dist(voxel.dist),
        grad(voxel.grad.normalized()),
        weight(voxel.weight),
        r(Vec8f::Zero()),
        g(Vec8f::Zero()),
        b(Vec8f::Zero())
    {
        const float voxel_size_4 = 0.25 * voxel_size;
        d[0] = dist + voxel_size_4 * (-grad[0] - grad[1] - grad[2]);
        d[1] = dist + voxel_size_4 * ( grad[0] - grad[1] - grad[2]);
        d[2] = dist + voxel_size_4 * (-grad[0] + grad[1] - grad[2]);
        d[3] = dist + voxel_size_4 * ( grad[0] + grad[1] - grad[2]);
        d[4] = dist + voxel_size_4 * (-grad[0] - grad[1] + grad[2]);
        d[5] = dist + voxel_size_4 * ( grad[0] - grad[1] + grad[2]);
        d[6] = dist + voxel_size_4 * (-grad[0] + grad[1] + grad[2]);
        d[7] = dist + voxel_size_4 * ( grad[0] + grad[1] + grad[2]);
    }

    SdfVoxelHr(const SdfVoxelHr &voxel) :
        dist(voxel.dist),
        grad(voxel.grad),
        weight(voxel.weight),
        d(voxel.d),
        r(voxel.r),
        g(voxel.g),
        b(voxel.b)
    {}

};

using Vec3i = Eigen::Vector3i;
using SdfLrMap = phmap::parallel_node_hash_map<Vec3i, SdfVoxel,   phmap::priv::hash_default_hash<Vec3i>, phmap::priv::hash_default_eq<Vec3i>, Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxel>>>;
using SdfHrMap = phmap::parallel_node_hash_map<Vec3i, SdfVoxelHr, phmap::priv::hash_default_hash<Vec3i>, phmap::priv::hash_default_eq<Vec3i>, Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxelHr>>>;

#endif // SDF_VOXEL_H_

