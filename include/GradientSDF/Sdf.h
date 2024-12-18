#ifndef SDF_H_
#define SDF_H_

// includes
#include <iostream>

namespace cv {
    template <typename T>
    class NormalEstimator;
}

/**
 * class declaration
 */
class Sdf {

protected:

// friends

    friend class RigidPointOptimizer;
    friend class RigidSdfOptimizer;

// variables
    
    float T_; // truncation distance in meters
    float inv_T_;
    size_t counter_; // frame counter

    float z_min_ = 0.5;
    float z_max_ = 3.5;
    
// methods
    
    float truncate(float sdf) const {
        return std::max(-T_, std::min(T_, sdf));
    }
    
    float weight(float sdf) const {
        float w = 0.f;
        if (sdf<=0.) {
            w = 1.f;
        }
        else if (sdf<=T_) {
            w = 1.f - sdf*inv_T_;
        }
        return w;
    }

    void increase_counter() {
        ++counter_;
    }

//    void init();
    
public:

// constructors / destructor
    
    Sdf() :
        T_(0.05),
        inv_T_(1./T_),
        counter_(0)
    {}
    
    Sdf(float T) :
        T_(T),
        inv_T_(1./T_),
        counter_(0)
    {}
    
    virtual ~Sdf() {}
    
// methods
    
    virtual float tsdf(Vec3f point, Vec3f* grad_ptr = nullptr) const = 0;

    virtual float weights(Vec3f point) const = 0;
    
    // virtual void update(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, const SE3 &pose, cv::NormalEstimator<float>* NEst = nullptr) = 0;
    
    // virtual void setup(const cv::Mat &color, const cv::Mat &depth, const Mat3f K, cv::NormalEstimator<float>* NEst = nullptr) {
    //     update(color, depth, K, SE3(), NEst);
    // }

    virtual void updateFromMesh(const std::vector<Eigen::Vector3f> &vertices,
                                const std::vector<Eigen::Vector3i> &faces) = 0;

    virtual void updateFromMesh1(const std::vector<Eigen::Vector3f> &vertices, 
                                const std::vector<Eigen::Vector3i> &faces,
                                const std::vector<Eigen::Vector3f> &vertex_normals) = 0;

    void set_zmin(float z_min) {
        z_min_ = z_min;
    }

    void set_zmax(float z_max) {
        z_max_ = z_max;
    }



// visualization / debugging

    virtual bool extract_mesh(std::string filename = "mesh.ply") {
        return false;
    }
    
    virtual bool extract_pc(std::string filename = "cloud.ply") {
        return false;
    }

    virtual bool save_sdf(std::string filename ="sdf.txt"){
        return false;
    }

    virtual bool load_sdf(std::string filename){
        return false;
    }

};

#endif // SDF_H_
