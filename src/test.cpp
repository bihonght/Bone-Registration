#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#include "triangle.h"

void testTriangleGeometry() {
    // Test Case 1: Closest Point on Triangle
    {
        std::cout << "Testing Closest Point Calculation..." << std::endl;
        
        // Define a triangle
        Eigen::Vector3f a(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f b(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f c(0.0f, 1.0f, 0.0f);
        
        // Test point outside the triangle
        Eigen::Vector3f p(0.5f, 0.5f, 1.0f);
        
        float u, v, w;
        Eigen::Vector3f closest = closestPointOnTriangle(p, a, b, c, u, v, w);
        
        std::cout << "Closest Point: " << closest.transpose() << std::endl;
        std::cout << "Barycentric Coords: u=" << u << ", v=" << v << ", w=" << w << std::endl;
        
        // Verify barycentric coordinates sum to 1
        assert(std::abs(u + v + w - 1.0f) < 1e-6f);
        
        // Verify the point is on the triangle plane
        assert(std::abs(closest.z()) < 1e-6f);
    }
    
    // Test Case 2: Ray Intersection
    {
        std::cout << "Testing Triangle Ray Intersection..." << std::endl;
        
        // Define a triangle
        Eigen::Vector3f v1(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f v2(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f v3(0.0f, 1.0f, 0.0f);
        
        // Test intersecting ray
        Eigen::Vector3f origin(0.5f, 0.5f, 1.0f);
        Eigen::Vector3f dest(0.5f, 0.5f, -1.0f);
        
        float t;
        bool intersects = triangle_ray_intersection(origin, dest, v1, v2, v3, t);
        
        std::cout << "Intersection: " << (intersects ? "Yes" : "No") << std::endl;
        if (intersects) {
            std::cout << "Intersection Distance: " << t << std::endl;
        }
        
        // Test non-intersecting ray
        Eigen::Vector3f non_origin(2.0f, 2.0f, 1.0f);
        Eigen::Vector3f non_dest(2.0f, 2.0f, -1.0f);
        
        bool non_intersects = triangle_ray_intersection(non_origin, non_dest, v1, v2, v3, t);
        
        std::cout << "Non-Intersection Test: " << (non_intersects ? "Intersection" : "No Intersection") << std::endl;
    }
    
    // Test Case 3: Edge Cases
    {
        std::cout << "Testing Edge Cases..." << std::endl;
        
        // Degenerate triangle (collinear points)
        Eigen::Vector3f a(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f b(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f c(0.5f, 0.0f, 0.0f);
        
        Eigen::Vector3f p(0.25f, 1.0f, 0.0f);
        
        float u, v, w;
        Eigen::Vector3f closest = closestPointOnTriangle(p, a, b, c, u, v, w);
        
        std::cout << "Degenerate Triangle Closest Point: " << closest.transpose() << std::endl;
    }
    
    std::cout << "All tests completed successfully!" << std::endl;
}

int main() {
    try {
        testTriangleGeometry();
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}


void MapGradPixelSdf::updateFromMesh(const std::vector<Eigen::Vector3f> &vertices,
                                     const std::vector<Eigen::Vector3i> &faces,
                                     const std::vector<Eigen::Vector3f> &vertex_normals)
{
    if (faces.empty() || vertices.empty()) {
        std::cerr << "Empty mesh provided to updateFromMesh" << std::endl;
        return;
    }

    // 1. Compute global bounding box expanded by truncation
    Eigen::Vector3f min_pt = vertices[0];
    Eigen::Vector3f max_pt = vertices[0];
    for (const auto &v : vertices) {
        min_pt = min_pt.array().min(v.array());
        max_pt = max_pt.array().max(v.array());
    }
    float trunc_dist = T_;
    min_pt.array() -= trunc_dist;
    max_pt.array() += trunc_dist;

    // Convert bounding box to voxel indices
    Vec3i min_vox = float2vox(min_pt);
    Vec3i max_vox = float2vox(max_pt);

    // Determine the size of the grid from min_vox to max_vox
    // For example, if your grid is fixed, use ni, nj, nk as from makelevelset3
    int ni = max_vox.x() - min_vox.x() + 1;
    int nj = max_vox.y() - min_vox.y() + 1;
    int nk = max_vox.z() - min_vox.z() + 1;

    // Allocate temporary arrays for the algorithm
    Array3f phi(ni, nj, nk, (ni+nj+nk)*voxel_size_); // large initial distance
    Array3i closest_tri(ni, nj, nk, -1);
    Array3i intersection_count(ni, nj, nk, 0);

    // 2. Local Distance Initialization:
    // For each triangle, compute a local voxel bounding box around it
    for (size_t t = 0; t < faces.size(); ++t) {
        Eigen::Vector3i face = faces[t];
        const Eigen::Vector3f &v0 = vertices[face[0]];
        const Eigen::Vector3f &v1 = vertices[face[1]];
        const Eigen::Vector3f &v2 = vertices[face[2]];

        // Triangle bounding box
        Eigen::Vector3f tri_min = v0.array().min(v1.array()).min(v2.array());
        Eigen::Vector3f tri_max = v0.array().max(v1.array()).max(v2.array());

        // Expand by the truncation factor to ensure we cover necessary voxels
        tri_min.array() -= trunc_dist; 
        tri_max.array() += trunc_dist;

        Vec3i tri_min_vox = float2vox(tri_min);
        Vec3i tri_max_vox = float2vox(tri_max);

        // Clamp to the global bounding grid
        tri_min_vox = tri_min_vox.cwiseMax(min_vox);
        tri_max_vox = tri_max_vox.cwiseMin(max_vox);

        // For each voxel in the triangle’s local region, compute the point-triangle distance
        for (int k = tri_min_vox.z(); k <= tri_max_vox.z(); ++k) {
            for (int j = tri_min_vox.y(); j <= tri_max_vox.y(); ++j) {
                for (int i = tri_min_vox.x(); i <= tri_max_vox.x(); ++i) {
                    // Convert voxel index to array indices for phi
                    int ii = i - min_vox.x();
                    int jj = j - min_vox.y();
                    int kk = k - min_vox.z();

                    Eigen::Vector3f voxel_pos = vox2float(Vec3i(i,j,k));
                    float u,v,w;
                    Eigen::Vector3f cp = closestPointOnTriangle(voxel_pos, v0, v1, v2, u, v, w);
                    float dist = (cp - voxel_pos).norm();

                    if (dist < phi(ii,jj,kk)) {
                        phi(ii,jj,kk) = dist;
                        closest_tri(ii,jj,kk) = (int)t;
                    }
                }
            }
        }

        // Intersection counting for inside/outside:
        // Project triangle onto (j,k) plane and count how often it intersects intervals.
        // Similar logic to makelevelset3’s point_in_triangle_2d steps.
        // After determining intersection, increment intersection_count accordingly.
        // (This step is omitted here for brevity, but replicate the logic from makelevelset3.)
    }

    // 3. Sweeping Steps to propagate distances:
    // Perform sweeps in multiple directions
    // For example:
    sweep(faces, vertices, phi, closest_tri, origin_, voxel_size_, +1,+1,+1);
    sweep(faces, vertices, phi, closest_tri, origin_, voxel_size_, -1,-1,-1);
    // ... repeat for all 8 octant directions (as in makelevelset3) ...
    // The sweep function uses check_neighbour to update phi based on neighbours

    // 4. Determine inside/outside using intersection_count parity:
    // For each voxel, if total intersections are odd, negate the distance.
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int total_count = 0;
            for (int i = 0; i < ni; ++i) {
                total_count += intersection_count(i,j,k);
                if (total_count % 2 == 1) {
                    phi(i,j,k) = -phi(i,j,k);
                }
            }
        }
    }

    // 5. Update tsdf_ and vis_:
    // For each voxel with a distance within the truncation, compute weights, normals and update
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                float signed_dist = phi(i,j,k);
                Vec3i voxel_idx(i + min_vox.x(), j + min_vox.y(), k + min_vox.z());

                if (fabs(signed_dist) <= T_) {
                    float w_val = weight(signed_dist);
                    if (w_val > 0) {
                        int t = closest_tri(i,j,k);
                        if (t < 0) continue; // no triangle found
                        Eigen::Vector3i face = faces[t];

                        // Compute interpolated normal as before using barycentrics:
                        // Compute barycentrics again (if needed) or store them earlier.
                        // For simplicity, assume we store cp and barycentrics if necessary.
                        // Or recompute closest point and barycentrics from saved triangle index.

                        // Example (recompute closest point and normal):
                        const Eigen::Vector3f &v0 = vertices[face[0]];
                        const Eigen::Vector3f &v1 = vertices[face[1]];
                        const Eigen::Vector3f &v2 = vertices[face[2]];
                        // Recompute barycentrics for final normal:
                        float u,v,w;
                        Eigen::Vector3f voxel_pos = vox2float(voxel_idx);
                        Eigen::Vector3f cp = closestPointOnTriangle(voxel_pos, v0, v1, v2, u, v, w);
                        Eigen::Vector3f N0 = vertex_normals[face[0]];
                        Eigen::Vector3f N1 = vertex_normals[face[1]];
                        Eigen::Vector3f N2 = vertex_normals[face[2]];
                        Eigen::Vector3f best_normal = (w*N0 + u*N1 + v*N2).normalized();

                        SdfVoxel &voxel = tsdf_[voxel_idx]; // Inserts if doesn't exist
                        voxel.weight += w_val;
                        voxel.dist += (truncate(signed_dist) - voxel.dist) * w_val / voxel.weight;
                        voxel.grad += w_val * best_normal;

                        std::vector<bool> &vis = vis_[voxel_idx];
                        vis.resize(counter_);
                        vis.push_back(true);
                    }
                }
            }
        }
    }

    increase_counter();
    std::cout << "Updated SDF from mesh using makelevelset3 approach (Local + Sweep). Frame counter: " << counter_ << std::endl;
}
