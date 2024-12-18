#ifndef MAT_H
#define MAT_H

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>
#include <sophus/se3.hpp>

typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4f Vec4f;
typedef Eigen::Matrix3f Mat3f;
typedef Eigen::Matrix4f Mat4f;

typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Matrix3d Mat3;
typedef Eigen::Matrix4d Mat4;

typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;

typedef Sophus::SE3<float> SE3;
typedef Sophus::SO3<float> SO3;

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

#include <algorithm> // for std::fill

template<class T>
inline T clamp(T a, T lower, T upper)
{
   if(a<lower) return lower;
   else if(a>upper) return upper;
   else return a;
}

// A 3D array of floats
class Array3f {
public:
    Array3f() : ni(0), nj(0), nk(0) {}
    Array3f(int ni_, int nj_, int nk_, float val=0.0f) {
        resize(ni_, nj_, nk_);
        assign(val);
    }
    void resize(int ni_, int nj_, int nk_) {
        ni = ni_; nj = nj_; nk = nk_;
        data.resize(ni * nj * nk, 0.0f);
    }
    void assign(float val) {
        std::fill(data.begin(), data.end(), val);
    }
    float& operator()(int i, int j, int k) {
        checkIndex(i,j,k);
        return data[i + ni*(j + nj*k)];
    }
    const float& operator()(int i, int j, int k) const {
        checkIndex(i,j,k);
        return data[i + ni*(j + nj*k)];
    }
    int getNi() const { return ni; }
    int getNj() const { return nj; }
    int getNk() const { return nk; }

private:
    void checkIndex(int i, int j, int k) const {
        if (i<0 || i>=ni || j<0 || j>=nj || k<0 || k>=nk)
            throw std::out_of_range("Array3f index out of range");
    }
    int ni, nj, nk;
    std::vector<float> data;
};

// A 3D array of ints
class Array3i {
public:
    Array3i() : ni(0), nj(0), nk(0) {}
    Array3i(int ni_, int nj_, int nk_, int val=0) {
        resize(ni_, nj_, nk_);
        assign(val);
    }
    void resize(int ni_, int nj_, int nk_) {
        ni = ni_; nj = nj_; nk = nk_;
        data.resize(ni * nj * nk, 0);
    }
    void assign(int val) {
        std::fill(data.begin(), data.end(), val);
    }
    int& operator()(int i, int j, int k) {
        checkIndex(i,j,k);
        return data[i + ni*(j + nj*k)];
    }
    const int& operator()(int i, int j, int k) const {
        checkIndex(i,j,k);
        return data[i + ni*(j + nj*k)];
    }
    int getNi() const { return ni; }
    int getNj() const { return nj; }
    int getNk() const { return nk; }

private:
    void checkIndex(int i, int j, int k) const {
        if (i<0 || i>=ni || j<0 || j>=nj || k<0 || k>=nk)
            throw std::out_of_range("Array3i index out of range");
    }
    int ni, nj, nk;
    std::vector<int> data;
};

class Array3Vec3f {
public:
    Array3Vec3f() : ni(0), nj(0), nk(0) {}
    Array3Vec3f(int ni_, int nj_, int nk_, const Eigen::Vector3f &val=Eigen::Vector3f::Zero()) {
        resize(ni_, nj_, nk_);
        assign(val);
    }
    void resize(int ni_, int nj_, int nk_) {
        ni = ni_; nj = nj_; nk = nk_;
        data.resize(ni * nj * nk, Eigen::Vector3f::Zero());
    }
    void assign(const Eigen::Vector3f &val) {
        std::fill(data.begin(), data.end(), val);
    }
    Eigen::Vector3f& operator()(int i, int j, int k) {
        checkIndex(i,j,k);
        return data[i + ni*(j + nj*k)];
    }
    const Eigen::Vector3f& operator()(int i, int j, int k) const {
        checkIndex(i,j,k);
        return data[i + ni*(j + nj*k)];
    }
    int getNi() const { return ni; }
    int getNj() const { return nj; }
    int getNk() const { return nk; }

private:
    void checkIndex(int i, int j, int k) const {
        if (i<0 || i>=ni || j<0 || j>=nj || k<0 || k>=nk)
            throw std::out_of_range("Array3Vec3f index out of range");
    }
    int ni, nj, nk;
    std::vector<Eigen::Vector3f> data;
};

#endif
