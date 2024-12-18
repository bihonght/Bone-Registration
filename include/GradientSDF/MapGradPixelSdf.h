#ifndef MAP_GRAD_PIXEL_SDF_H_
#define MAP_GRAD_PIXEL_SDF_H_

// includes
#include <iostream>
#include "mat.h"
// #include <opencv2/core/core.hpp>
// class includes
#include "SdfVoxel.h"
#include "Sdf.h"
#include "hash_map.h"
/**
 * class declaration
 */
class MapGradPixelSdf : public Sdf {

// friends

    friend class PhotometricOptimizer;

// variables

    const float voxel_size_;
    const float voxel_size_inv_;
 
    // phmap::parallel_flat_hash_map<Vec3i, SdfVoxel> tsdf_;
    phmap::parallel_node_hash_map<Eigen::Vector3i, SdfVoxel,
                phmap::priv::hash_default_hash<Eigen::Vector3i>,
                phmap::priv::hash_default_eq<Eigen::Vector3i>,
                Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, SdfVoxel>>> tsdf_;

    phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>> vis_;
    
// methods

    Vec3i float2vox(Vec3f point) const {
        Vec3f pv = voxel_size_inv_ * point;
        return Vec3i(std::round(pv[0]), std::round(pv[1]), std::round(pv[2]));
    }
    
    Vec3f vox2float(Vec3i idx) const {
        return voxel_size_ * idx.cast<float>();
    }
    
public:

// constructors / destructor
    
    MapGradPixelSdf() :
        Sdf(),
        voxel_size_(0.02),
        voxel_size_inv_(1./voxel_size_)
    {}
    
    MapGradPixelSdf(float voxel_size) :
        Sdf(),
        voxel_size_(voxel_size),
        voxel_size_inv_(1./voxel_size_)
    {}
    
    MapGradPixelSdf(float voxel_size, float T) :
        Sdf(T),
        voxel_size_(voxel_size),
        voxel_size_inv_(1./voxel_size_)
    {}
    
    ~MapGradPixelSdf() {}
    
// methods
    
    virtual float tsdf(Vec3f point, Vec3f* grad_ptr) const {
        const Vec3i idx = float2vox(point);
        const SdfVoxel& v = tsdf_.at(idx); // at performs bound checking, which is not necessary, but otherwise tsdf_ cannot be const
        if (grad_ptr)
            (*grad_ptr) = 1.2*v.grad.normalized(); // factor 1.2 corrects for SDF scaling due to projectiveness (heuristic)
        return v.dist + 1.2*v.grad.normalized().dot(vox2float(idx) - point);
    }
    
    virtual float weights(Vec3f point) const {
        const Vec3i idx = float2vox(point);
        auto pair = tsdf_.find(idx);
        if (pair != tsdf_.end()){
            // std::cout << idx << std::endl;
            return pair->second.weight;
        }
        return 0;
    }

    SdfVoxel getSdf(Vec3i idx) {
        return tsdf_.at(idx);
    }
    
    // virtual void update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst);

    virtual void updateFromMesh(const std::vector<Eigen::Vector3f> &vertices, const std::vector<Eigen::Vector3i> &faces);

    virtual void updateFromMesh1(const std::vector<Eigen::Vector3f> &vertices,
                               const std::vector<Eigen::Vector3i> &faces,
                               const std::vector<Eigen::Vector3f> &vertex_normals);
                            
    phmap::parallel_node_hash_map<Vec3i, SdfVoxel,
                phmap::priv::hash_default_hash<Vec3i>,
                phmap::priv::hash_default_eq<Vec3i>,
                Eigen::aligned_allocator<std::pair<const Vec3i, SdfVoxel>>> get_tsdf() const {
        return tsdf_;
    }

    phmap::parallel_flat_hash_map<Vec3i, std::vector<bool>>& get_vis() {
        return vis_;
    }   

// visualization / debugging
    
    virtual bool extract_mesh(std::string filename);
    
    virtual bool extract_pc(std::string filename);

    virtual bool save_sdf(std::string filename);

    virtual bool load_sdf(std::string filename);
};

#endif // MAP_GRAD_PIXEL_SDF_H_
