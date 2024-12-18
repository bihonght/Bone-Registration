#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
// #include <limits>

// Compute the closest point on a triangle (a, b, c) to a point p.
// Returns the closest point and barycentric coordinates u, v, w.
// Barycentric coordinates: 
//    closestPoint = a*u + b*v + c*w, with u+v+w=1
// If the closest point is inside the triangle, all u,v,w ≥ 0.
// If not, one or more are < 0, and we clamp to edges/vertices.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

Eigen::Vector3f closestPointOnTriangle(const Eigen::Vector3f &p,
                                       const Eigen::Vector3f &a,
                                       const Eigen::Vector3f &b,
                                       const Eigen::Vector3f &c,
                                       float &u, float &v, float &w)
{
    // Edges from a
    Eigen::Vector3f ab = b - a;
    Eigen::Vector3f ac = c - a;
    Eigen::Vector3f ap = p - a;
    // Compute dot products
    float d00 = ab.dot(ab); // |ab|²
    float d01 = ab.dot(ac);
    float d11 = ac.dot(ac); // |ac|²
    float d20 = ap.dot(ab);
    float d21 = ap.dot(ac);
    // Compute denominator of barycentric formula
    float denom = d00 * d11 - d01 * d01;
    if (std::fabs(denom) < 1e-12f) {
        // Degenerate triangle: all vertices might be collinear.
        // Just pick the closest among the vertices a, b, c.
        float da = (p - a).squaredNorm();
        float db = (p - b).squaredNorm();
        float dc = (p - c).squaredNorm();

        if (da <= db && da <= dc) {
            u = 1.0f; v = 0.0f; w = 0.0f; return a;
        } else if (db <= dc) {
            u = 0.0f; v = 1.0f; w = 0.0f; return b;
        } else {
            u = 0.0f; v = 0.0f; w = 1.0f; return c;
        }
    }
    float invDenom = 1.0f / denom;
    float u_bar = (d11 * d20 - d01 * d21) * invDenom;
    float v_bar = (d00 * d21 - d01 * d20) * invDenom;
    float w_bar = 1.0f - u_bar - v_bar;
    // If inside the triangle, all barycentrics ≥ 0
    if (u_bar >= 0.0f && v_bar >= 0.0f && w_bar >= 0.0f) {
        u = u_bar; v = v_bar; w = w_bar;
        return a + u_bar * ab + v_bar * ac;
    }
    // Otherwise, the closest point lies on an edge or vertex.
    // Check each possibility:
    // Edges are:
    //    AB (w=0), 
    //    BC (u=0), 
    //    CA (v=0).

    // Compute projections onto each edge and find the closest one
    // Edge AB
    Eigen::Vector3f cpAB;
    float tAB = ap.dot(ab) / d00;
    tAB = std::fmax(0.0f, std::fmin(1.0f, tAB));
    cpAB = a + tAB * ab;
    float distAB = (p - cpAB).squaredNorm();

    // Edge BC
    Eigen::Vector3f bc = c - b;
    Eigen::Vector3f bp = p - b;
    float dBC = bc.dot(bc);
    float tBC = bp.dot(bc) / dBC;
    tBC = std::fmax(0.0f, std::fmin(1.0f, tBC));
    Eigen::Vector3f cpBC = b + tBC * bc;
    float distBC = (p - cpBC).squaredNorm();

    // Edge CA
    Eigen::Vector3f ca = a - c;
    Eigen::Vector3f cp = p - c;
    float dCA = ca.dot(ca);
    float tCA = cp.dot(ca) / dCA;
    tCA = std::fmax(0.0f, std::fmin(1.0f, tCA));
    Eigen::Vector3f cpCA = c + tCA * ca;
    float distCA = (p - cpCA).squaredNorm();

    // Determine the smallest distance
    if (distAB <= distBC && distAB <= distCA) {
        u = 1.0f - tAB;
        v = tAB;
        w = 0.0f;
        return cpAB;
    } else if (distBC <= distAB && distBC <= distCA) {
        u = 0.0f;
        v = 1.0f - tBC;
        w = tBC;
        return cpBC;
    } else {
        u = tCA;
        v = 0.0f;
        w = 1.0f - tCA;
        return cpCA;
    }
}


bool intersect_triangle(const Eigen::Vector3f &ray_origin,
                               const Eigen::Vector3f &ray_dir,
                               const Eigen::Vector3f &v0,
                               const Eigen::Vector3f &v1,
                               const Eigen::Vector3f &v2,
                               float &t) 
{
    // Möller–Trumbore intersection algorithm
    
    const float EPSILON = 1e-8f;
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;

    Eigen::Vector3f pvec = ray_dir.cross(edge2);
    float det = edge1.dot(pvec);

    // If det is near zero, ray lies in plane of triangle or is parallel to it
    if (std::fabs(det) < EPSILON) {
        return false;
    }

    float invDet = 1.0f / det;
    Eigen::Vector3f tvec = ray_origin - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    Eigen::Vector3f qvec = tvec.cross(edge1);
    float v = ray_dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    // t is the distance along the ray where intersection occurs
    float temp_t = edge2.dot(qvec) * invDet;
    if (temp_t > EPSILON) {
        t = temp_t;
        return true;
    }

    return false;
}

bool triangle_ray_intersection(const Eigen::Vector3f &origin, const Eigen::Vector3f &dest, 
  const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const Eigen::Vector3f &v3, float &t) {

    Eigen::Vector3f dir = (dest - origin).normalized();
    bool success = intersect_triangle(origin, dir, v1, v2, v3, t);
    if (success) {
        return true;
    }
    
    return false;
}

int orientation(float x1, float y1, float x2, float y2, float &twice_signed_area)
{
   twice_signed_area=y1*x2-x1*y2;
   if(twice_signed_area>0) return 1;
   else if(twice_signed_area<0) return -1;
   else if(y2>y1) return 1;
   else if(y2<y1) return -1;
   else if(x1>x2) return 1;
   else if(x1<x2) return -1;
   else return 0; // only true when x1==x2 and y1==y2
}

// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
// if true is returned, the barycentric coordinates are set in a,b,c.
bool point_in_triangle_2d(float x0, float y0, 
                                 float x1, float y1, float x2, float y2, float x3, float y3,
                                 float& a, float& b, float& c)
{
   x1-=x0; x2-=x0; x3-=x0;
   y1-=y0; y2-=y0; y3-=y0;
   int signa=orientation(x2, y2, x3, y3, a);
   if(signa==0) return false;
   int signb=orientation(x3, y3, x1, y1, b);
   if(signb!=signa) return false;
   int signc=orientation(x1, y1, x2, y2, c);
   if(signc!=signa) return false;
   double sum=a+b+c;
   assert(sum!=0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
   a/=sum;
   b/=sum;
   c/=sum;
   return true;
}

Eigen::Vector3f closestPointTriangle(Eigen::Vector3f const& p, Eigen::Vector3f const& a, Eigen::Vector3f const& b, Eigen::Vector3f const& c)
  {
    const Eigen::Vector3f ab = b - a;
    const Eigen::Vector3f ac = c - a;
    const Eigen::Vector3f ap = p - a;

    const float d1 = ab.dot(ap);
    const float d2 = ac.dot(ap);
    if (d1 <= 0.f && d2 <= 0.f) return a;

    const Eigen::Vector3f bp = p - b;
    const float d3 = ab.dot(bp);
    const float d4 = ac.dot(bp);
    if (d3 >= 0.f && d4 <= d3) return b;

    const Eigen::Vector3f cp = p - c;
    const float d5 = ab.dot(cp);
    const float d6 = ac.dot(cp);
    if (d6 >= 0.f && d5 <= d6) return c;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
    {
        const float v = d2 / (d2 - d6);
        return a + v * ac;
    }
    
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
    {
        const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + v * (c - b);
    }

    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + v * ab + w * ac;
  }

static void check_neighbour(const std::vector<Vec3i> &tri, const std::vector<Vec3f> &x,
                            Array3f &phi, Array3i &closest_tri,
                            const Vec3f &gx, int i0, int j0, int k0, int i1, int j1, int k1)
{
   if(closest_tri(i1,j1,k1)>=0){
    unsigned int face = closest_tri(i1,j1,k1);
    unsigned int p=tri[face][0], q=tri[face][1], r=tri[face][2]; 
    float d=(closestPointTriangle(gx, x[p], x[q], x[r]) - gx).norm(); //
    if(d<phi(i0,j0,k0)){
        phi(i0,j0,k0)=d;
        closest_tri(i0,j0,k0)=closest_tri(i1,j1,k1);
    }
   }
}

static void sweep(const std::vector<Vec3i> &tri, const std::vector<Vec3f> &x,
                  Array3f &phi, Array3i &closest_tri, const Vec3f &origin, float dx,
                  int di, int dj, int dk)
{
   int i0, i1;
   if(di>0){ i0=1; i1=phi.getNi(); }
   else{ i0=phi.getNi()-2; i1=-1; }
   int j0, j1;
   if(dj>0){ j0=1; j1=phi.getNj(); }
   else{ j0=phi.getNj()-2; j1=-1; }
   int k0, k1;
   if(dk>0){ k0=1; k1=phi.getNk(); }
   else{ k0=phi.getNk()-2; k1=-1; }
   for(int k=k0; k!=k1; k+=dk) for(int j=j0; j!=j1; j+=dj) for(int i=i0; i!=i1; i+=di){
      Vec3f gx(i*dx+origin[0], j*dx+origin[1], k*dx+origin[2]);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i-di, j,    k);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i,    j-dj, k);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i-di, j-dj, k);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i,    j,    k-dk);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i-di, j,    k-dk);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i,    j-dj, k-dk);
      check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i-di, j-dj, k-dk);
   }
}

#endif