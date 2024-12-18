#include "MapGradPixelSdf.h"
// #include "normals/NormalEstimator.h"
#include "mesh/LayeredMarchingCubesNoColor.h"
#include "triangle.h" 

#include <fstream>

void MapGradPixelSdf::updateFromMesh(const std::vector<Eigen::Vector3f> &vertices, 
                                     const std::vector<Eigen::Vector3i> &faces) {
    const float z_min = z_min_, z_max = z_max_;
    const int factor = std::floor(T_ / voxel_size_);

    for (const auto& face : faces) {
        Eigen::Vector3f v0 = vertices[face[0]];
        Eigen::Vector3f v1 = vertices[face[1]];
        Eigen::Vector3f v2 = vertices[face[2]];

        // Calculate normal
        Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();

        // Instead of sampling just from v0 along the normal, let's sample a small grid
        // inside the triangle and then move along the normal direction for each point.
        // 'steps' determines how finely we sample inside the triangle.
        int steps = 5;  // Increase this for finer sampling
        for (int iu = 0; iu <= steps; ++iu) {
            for (int iv = 0; iv <= steps - iu; ++iv) {
                float u = static_cast<float>(iu) / steps;
                float w = static_cast<float>(iv) / steps;
                // Point inside the triangle using barycentric interpolation
                Eigen::Vector3f point_in_triangle = v0 + u * (v1 - v0) + w * (v2 - v0);

                // Now sample along the normal direction from this interior point
                for (int k = -factor; k <= factor; ++k) {
                    Eigen::Vector3f point = point_in_triangle + k * voxel_size_ * normal;

                    // Convert to voxel coordinates
                    Vec3i voxel_idx = float2vox(point);
                    Eigen::Vector3f voxel_pos = vox2float(voxel_idx);

                    // Compute signed distance along the normal direction
                    float sdf = normal.dot(voxel_pos - point_in_triangle);
                    float weight_val = weight(sdf);

                    if (weight_val > 0) {
                        SdfVoxel &voxel = tsdf_[voxel_idx];
                        // Weighted update of distance
                        voxel.weight += weight_val;
                        voxel.dist += (truncate(sdf) - voxel.dist) * weight_val / voxel.weight;
                        voxel.grad += weight_val * normal;

                        std::vector<bool> &vis = vis_[voxel_idx];
                        if (vis.size() < (size_t)counter_ + 1) {
                            // Ensure vis can hold current frame visibility
                            vis.resize(counter_ + 1, false);
                        }
                        vis[counter_] = true;
                    }
                }
            }
        }
    }

    increase_counter();
    std::cout << "Updated SDF from mesh. Frame counter: " << counter_ << std::endl;
}

void MapGradPixelSdf::updateFromMesh1(const std::vector<Eigen::Vector3f> &vertices, const std::vector<Eigen::Vector3i> &faces, const std::vector<Eigen::Vector3f> &vertex_normals)
{
    if (faces.empty() || vertices.empty()) {
        std::cerr << "Empty mesh provided to updateFromMesh" << std::endl;
        return;
    }

    // Compute bounding box of the mesh
    Eigen::Vector3f min_pt = vertices[0];
    Eigen::Vector3f max_pt = vertices[0];
    for (const auto &v : vertices) {
        min_pt = min_pt.array().min(v.array());
        max_pt = max_pt.array().max(v.array());
    }
    std::cout << "Before Expand min_pt = " << min_pt << "\n max_pt = " << max_pt << std::endl; 
    // Expand bounding box by truncation distance
    float trunc_dist = T_;
    int trunc_factor = 2; //std::floor(T_ * voxel_size_inv_); 
    min_pt.array() -= trunc_dist;
    max_pt.array() += trunc_dist;
    Eigen::Vector3f origin_ = min_pt; 
    
    std::cout << "After Expand min_pt = " << min_pt << "\n max_pt = " << max_pt << std::endl; 
    // Convert bounding box to voxel indices
    Vec3i min_vox = float2vox(min_pt);
    Vec3i max_vox = float2vox(max_pt);
    std::cout << "min_vox = " << ": [" << min_vox[0] << ", " << min_vox[1] << ", " << min_vox[2] << "]" << std::endl; 
    std::cout << "max_vox = " << ": [" << max_vox[0] << ", " << max_vox[1] << ", " << max_vox[2] << "]" << std::endl;

    std::cout << "Start = " << std::endl;

    // Determine the size of the grid from min_vox to max_vox
    int ni = max_vox.x() - min_vox.x() + 1;
    int nj = max_vox.y() - min_vox.y() + 1;
    int nk = max_vox.z() - min_vox.z() + 1;

    // Allocate temporary arrays for the algorithm
    Array3f phi(ni, nj, nk, (ni+nj+nk)*voxel_size_); // large initial distance
    Array3i closest_tri(ni, nj, nk, -1);
    Array3i intersection_count(ni, nj, nk, 0);
    Array3Vec3f closest_norms(ni, nj, nk, Eigen::Vector3f::Zero());
    Array3i weight_vals(ni, nj, nk, -1);
    
    std::cout << ("done allocate temporary array") << std::endl;
    for (size_t t = 0; t < faces.size(); ++t) {
        if (t%2 == 0) std::cout << (100*t/faces.size()) << std::endl;
        Eigen::Vector3i face = faces[t];
        const Eigen::Vector3f &v0 = vertices[face[0]];
        const Eigen::Vector3f &v1 = vertices[face[1]];
        const Eigen::Vector3f &v2 = vertices[face[2]];

        // Triangle bounding box
        Eigen::Vector3f tri_min = v0.array().min(v1.array()).min(v2.array());
        Eigen::Vector3f tri_max = v0.array().max(v1.array()).max(v2.array());
        // Convert to voxel coordinates 
        Vec3i tri_min_vox = float2vox(tri_min);
        Vec3i tri_max_vox = float2vox(tri_max);
        // Expand by the truncation factor to ensure we cover necessary voxels
        tri_min_vox.array() -= trunc_factor; 
        tri_max_vox.array() += trunc_factor;
        // Clamp to the global bounding grid
        tri_min_vox = tri_min_vox.cwiseMax(min_vox);
        tri_max_vox = tri_max_vox.cwiseMin(max_vox);

        // For each voxel in the triangle’s local region, compute the point-triangle distance
        for (int k = tri_min_vox.z(); k <= tri_max_vox.z(); ++k) {
            for (int j = tri_min_vox.y(); j <= tri_max_vox.y(); ++j) {
                for (int i = tri_min_vox.x(); i <= tri_max_vox.x(); ++i) {
                    // Convert voxel index to array indices for phi
                    int ii = clamp(i - min_vox.x(), 0, ni-1);
                    int jj = clamp(j - min_vox.y(), 0, nj-1);
                    int kk = clamp(k - min_vox.z(), 0, nk-1);
                    
                    Eigen::Vector3f voxel_pos = vox2float(Vec3i(i,j,k));
                    float u,v,w;
                    Eigen::Vector3f cp = closestPointTriangle(voxel_pos, v0, v1, v2); //closestPointOnTriangle(voxel_pos, v0, v1, v2, u, v, w);
                    float dist = (cp - voxel_pos).norm();
                    // Eigen::Vector3f N0 = vertex_normals[face[0]];
                    // Eigen::Vector3f N1 = vertex_normals[face[1]];
                    // Eigen::Vector3f N2 = vertex_normals[face[2]];
                    // Eigen::Vector3f interp_normal = (w*N0 + u*N1 + v*N2).normalized();
                    Eigen::Vector3f interp_normal = (voxel_pos - cp).normalized();
                    
                    float weight_val = weight(dist);
                    if (dist < phi(ii,jj,kk)) {
                        weight_vals(ii,jj,kk) = weight_val;
                        phi(ii,jj,kk) = dist;
                        closest_tri(ii,jj,kk) = (int)t;
                        closest_norms(ii,jj,kk) = weight_val*interp_normal;
                    }
                    else if (dist == phi(ii,jj,kk)) {
                        weight_vals(ii,jj,kk) += weight_val;
                        phi(ii,jj,kk) = dist;
                        closest_tri(ii,jj,kk) = (int)t;
                        closest_norms(ii,jj,kk) += weight_val * interp_normal;
                    }
                }
            }
        }

        // Update intersection counts
        // Return back to original voxel boundary conditions
        tri_min_vox.array() += trunc_factor; 
        tri_max_vox.array() -= trunc_factor;
        tri_min_vox = tri_min_vox.cwiseMax(min_vox);
        tri_max_vox = tri_max_vox.cwiseMin(max_vox);
        // coordinates in voxel grid
        Vec3i v0_vox = float2vox(v0), v1_vox = float2vox(v1), v2_vox = float2vox(v2);

        for (int k = tri_min_vox.z(); k <= tri_max_vox.z(); ++k) {
            for (int j = tri_min_vox.y(); j <= tri_max_vox.y(); ++j) {
                int jj = clamp(j - min_vox.y(), 0, nj-1);
                int kk = clamp(k - min_vox.z(), 0, nk-1);
                float a, b, c; // bayer index
                Eigen::Vector3f voxel_pos = vox2float(Vec3i(0,j,k));
                if (point_in_triangle_2d(voxel_pos.y(), voxel_pos.z(), v0.y(), v0.z(), v1.y(), v1.z(), v2.y(), v2.z(), a, b, c)) {
                    voxel_pos.x() = a*v0.x() + b*v1.x() + c*v2.x();
                    int i = std::ceil(voxel_pos.x()*voxel_size_inv_);

                    int ii = i - min_vox.x();
                    if(ii<0) ++intersection_count(0, jj, kk); // we enlarge the first interval to include everything to the -x direction
                    else if(ii<ni) ++intersection_count(ii,jj,kk);
                    // we ignore intersections that are beyond the +x side of the grid
                }
            }
        }
    }

    // and now we fill in the rest of the distances with fast sweeping
    for(unsigned int pass=0; pass<2; ++pass){
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, +1, +1, +1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, -1, -1, -1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, +1, +1, -1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, -1, -1, +1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, +1, -1, +1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, -1, +1, -1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, +1, -1, -1);
        sweep(faces, vertices, phi, closest_tri, min_pt, voxel_size_, -1, +1, +1);
    }

    std::cout << "Hollaa " << std::endl;
    // Perform signed distance field computation
    for (int k = min_vox.z(); k <= max_vox.z(); ++k) {
        for (int j = min_vox.y(); j <= max_vox.y(); ++j) {
            int total_count = 0;
            for (int i = min_vox.x(); i <= max_vox.x(); ++i) {
                // Convert voxel index to array indices for phi
                int ii = i - min_vox.x();
                int jj = j - min_vox.y();
                int kk = k - min_vox.z();

                total_count += intersection_count(ii,jj,kk); // update intersection count for signed distance
                if (total_count % 2 == 1) {
                    phi(ii,jj,kk)*=-1; 
                    closest_norms(ii,jj,kk)*=-1;
                }
                float sdf = phi(ii,jj,kk); 
                Eigen::Vector3i voxel_idx(i,j,k);
                Eigen::Vector3f normal = closest_norms(ii,jj,kk);
                float weight_val = weight(sdf);
                // store voxel data
                if (weight_val > 0) {
                    SdfVoxel& voxel = tsdf_[voxel_idx];
                    voxel.weight += weight_val;
                    // voxel.dist += (truncate(sdf) - voxel.dist) * weight_val / voxel.weight;
                    voxel.dist += (sdf - voxel.dist) * weight_val / voxel.weight;
                    voxel.grad += normal; // voxel.grad += weight_val * normal;

                    std::vector<bool>& vis = vis_[voxel_idx];
                    vis.resize(counter_);
                    vis.push_back(true);
                }
            }
        }
    }    
    increase_counter();
    std::cout << "Updated SDF from mesh (without BVH). Frame counter: " << counter_ << std::endl;
}

bool MapGradPixelSdf::extract_mesh(std::string filename) {

    // compute dimensions (and, from that, size)
    const int pos_inf = std::numeric_limits<int>::max();
    const int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;
    for (auto v : tsdf_) {
        if (v.first[0] < xmin) xmin = v.first[0];
        if (v.first[0] > xmax) xmax = v.first[0];
        if (v.first[1] < ymin) ymin = v.first[1];
        if (v.first[1] > ymax) ymax = v.first[1];
        if (v.first[2] < zmin) zmin = v.first[2];
        if (v.first[2] > zmax) zmax = v.first[2];
    }

    // create input that can handle MarchingCubes class
    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    std::cout << "dimension " << dim[0] << " * " << dim[1] << " * " << dim[2] << std::endl;

    const size_t num_voxels = static_cast<size_t>(dim[0]) *
                            static_cast<size_t>(dim[1]) *
                            static_cast<size_t>(dim[2]);
    float* dist = new float[num_voxels];
    float* weights = new float[num_voxels];
    size_t lin_index = 0;
    for (int k=zmin; k<=zmax; ++k) for (int j=ymin; j<=ymax; ++j) for (int i=xmin; i<=xmax; ++i) {
        Vec3i idx(i, j, k);
        auto pair = tsdf_.find(idx);
        if (pair != tsdf_.end()) {
            dist[lin_index] = pair->second.dist;
            weights[lin_index] = pair->second.weight;
        }
        else {
            dist[lin_index] = T_;
            weights[lin_index] = 0;
        }
        ++lin_index;
    }
    
    // call marching cubes
    LayeredMarchingCubesNoColor lmc(Vec3f(voxel_size_, voxel_size_, voxel_size_));
    lmc.computeIsoSurface(&tsdf_);
    bool success = lmc.savePly(filename);
    
    // delete temporary arrays  
    delete[] dist;
    delete[] weights;
    
    return success;
}

bool MapGradPixelSdf::extract_pc(std::string filename) {

	const float voxel_size_2 = .5 * voxel_size_;

    std::vector<Vec6f> points_normals;
    for (const auto& el : tsdf_) {
        const SdfVoxel& v = el.second;
        if (v.weight < 0.2)  /// 5
            continue;
        Vec3f g = 1.2*v.grad.normalized();
        Vec3f d = v.dist * g;
        // if (std::fabs(d[0]) < voxel_size_2 && std::fabs(d[1]) < voxel_size_2 && std::fabs(d[2]) < voxel_size_2)
        if (std::fabs(d[0]) < voxel_size_2 && std::fabs(d[1]) < voxel_size_2 && std::fabs(d[2]) < voxel_size_2 && std::fabs(v.dist) < voxel_size_)
        {
            Vec6f pn;
            pn.segment<3>(0) = vox2float(el.first) - d;
            pn.segment<3>(3) = -g;
            points_normals.push_back(pn);
        }
    }    

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;
        
    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << points_normals.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property float nx" << std::endl;
    plyFile << "property float ny" << std::endl;
    plyFile << "property float nz" << std::endl;
    plyFile << "end_header" << std::endl;
    
    for (const Vec6f& p : points_normals) {
        plyFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
    }
    
    plyFile.close();

    return true;
}

bool MapGradPixelSdf::save_sdf(std::string filename)
{
    // compute dimensions (and, from that, size)
    const int pos_inf = std::numeric_limits<int>::max();
    const int neg_inf = std::numeric_limits<int>::min();
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin = pos_inf;
    xmax = neg_inf;
    ymin = pos_inf;
    ymax = neg_inf;
    zmin = pos_inf;
    zmax = neg_inf;
    for (auto v : tsdf_) {
        if (v.first[0] < xmin) xmin = v.first[0];
        if (v.first[0] > xmax) xmax = v.first[0];
        if (v.first[1] < ymin) ymin = v.first[1];
        if (v.first[1] > ymax) ymax = v.first[1];
        if (v.first[2] < zmin) zmin = v.first[2];
        if (v.first[2] > zmax) zmax = v.first[2];
    }

    // create input that can handle MarchingCubes class
    const Vec3i dim(xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1);
    const size_t num_voxels = dim[0] * dim[1] * dim[2];

    // Create a subdirectory to store all the files
    std::string folder = filename + "_data";
    std::filesystem::create_directories(folder);

    // --------------save grid info ----------------------------
    {
        std::ofstream grid_file(folder + "/grid_info.txt");
        if (!grid_file.is_open()) {
            std::cerr << "couldn't save grid_info file!" << std::endl;
            return false;
        }
        grid_file << "voxel size: " << voxel_size_ << std::endl 
                  << "voxel dim: " << dim[0] << " " << dim[1] << " " << dim[2] << std::endl
                  << "voxel min: " << xmin << " " << ymin << " " << zmin << std::endl
                  << "voxel max: " << xmax << " " << ymax << " " << zmax << std::endl;
    }

    // ---------------- save sdf dist and weight -------------------------------
    std::ofstream file(folder + "/sdf_d.txt");
    std::ofstream weight_file(folder + "/sdf_weight.txt");

    // -------------- save sdf grad ------------
    std::ofstream sdf_n0(folder + "/sdf_n0.txt");
    std::ofstream sdf_n1(folder + "/sdf_n1.txt");
    std::ofstream sdf_n2(folder + "/sdf_n2.txt");

    if (!file.is_open() || !weight_file.is_open() || !sdf_n0.is_open() || !sdf_n1.is_open() || !sdf_n2.is_open()){
        std::cerr << "couldn't save one of the sdf files!" << std::endl;
        return false;
    }

    for(const auto& pair : tsdf_){
        Vec3i idx = pair.first;
        const SdfVoxel& v = pair.second;
        int lin_idx = dim[0]*dim[1]*(idx[2]-zmin) + dim[0]*(idx[1]-ymin) + (idx[0]-xmin);
        file << lin_idx << " " << v.dist << "\n";
        weight_file << lin_idx << " " << v.weight << "\n";
        sdf_n0 << lin_idx << " " << v.grad[0] << "\n";
        sdf_n1 << lin_idx << " " << v.grad[1] << "\n";
        sdf_n2 << lin_idx << " " << v.grad[2] << "\n";
    }

    return true;
}
bool MapGradPixelSdf::load_sdf(std::string filename) {
    // Construct folder name
    std::string folder = filename + "_data";
    // Open grid info file from folder
    std::ifstream grid_file((folder + "/grid_info.txt").c_str());
    if (!grid_file.is_open()) {
        std::cerr << "Couldn't open grid_info file for loading!" << std::endl;
        return false;
    }

    int dim_x, dim_y, dim_z;
    int xmin, ymin, zmin, xmax, ymax, zmax;
    {
        std::string line;
        // Adjust the parsing logic according to how you saved the file.
        // Assuming the file format is as in your previous code snippet:
        // voxel size: <float>
        // voxel dim: dx dy dz
        // voxel min: xmin ymin zmin
        // voxel max: xmax ymax zmax

        // Skip voxel size line
        if (!std::getline(grid_file, line)) return false;
        // The next line should be voxel dim:
        if (!std::getline(grid_file, line)) return false; 
        {
            std::istringstream iss(line);
            std::string dummy;
            // e.g. "voxel dim: 100 120 130"
            iss >> dummy >> dummy;
            iss >> dim_x >> dim_y >> dim_z;
        }
        // Next line: voxel min
        if (!std::getline(grid_file, line)) return false;
        {
            std::istringstream iss(line);
            std::string dummy;
            // e.g. "voxel min: 10 20 30"
            iss >> dummy >> dummy;
            iss >> xmin >> ymin >> zmin;
        }
        // Next line: voxel max
        if (!std::getline(grid_file, line)) return false;
        {
            std::istringstream iss(line);
            std::string dummy;
            // e.g. "voxel max: 110 140 160"
            iss >> dummy >> dummy;
            iss >> xmax >> ymax >> zmax;
        }
    }

    grid_file.close();

    const int size_x = dim_x;
    const int size_y = dim_y;
    const int size_z = dim_z;
    const size_t num_voxels = static_cast<size_t>(size_x)*size_y*size_z;
    // Prepare temporary storage
    std::vector<float> dist_data(num_voxels, 0.0f);
    std::vector<float> weight_data(num_voxels, 0.0f);
    std::vector<float> grad_n0(num_voxels, 0.0f);
    std::vector<float> grad_n1(num_voxels, 0.0f);
    std::vector<float> grad_n2(num_voxels, 0.0f);
    // Helper lambda to load lin_idx-value pairs
    auto load_file = [&](const std::string &fname, std::vector<float> &out) {
        std::ifstream in(fname.c_str());
        if (!in.is_open()) {
            std::cerr << "Couldn't open " << fname << " for loading!" << std::endl;
            return false;
        }
        int lin_idx;
        float val;
        while (in >> lin_idx >> val) {
            if (lin_idx < 0 || static_cast<size_t>(lin_idx) >= num_voxels) {
                std::cerr << "lin_idx out of range in " << fname << std::endl;
                return false;
            }
            out[lin_idx] = val;
        }
        in.close();
        return true;
    };
    // Load each SDF-related file from the folder
    if (!load_file(folder + "/sdf_d.txt", dist_data)) return false;
    if (!load_file(folder + "/sdf_weight.txt", weight_data)) return false;
    if (!load_file(folder + "/sdf_n0.txt", grad_n0)) return false;
    if (!load_file(folder + "/sdf_n1.txt", grad_n1)) return false;
    if (!load_file(folder + "/sdf_n2.txt", grad_n2)) return false;

    tsdf_.clear();
    vis_.clear();
    // Reconstruct tsdf_ and vis_
    // lin_idx = (x - xmin) + (y - ymin)*size_x + (z - zmin)*size_x*size_y
    std::cout << "LOADING SDF " << std::endl;

    for (size_t i = 0; i < num_voxels; i++) {
        int i_int = static_cast<int>(i);

        int z = zmin + (i_int / (size_x * size_y));
        int remainder = i_int % (size_x * size_y);
        int y = ymin + (remainder / size_x);
        int x = xmin + (remainder % size_x);

        Vec3i idx(x, y, z);
        SdfVoxel voxel;
        voxel.dist = dist_data[i];
        voxel.weight = weight_data[i];
        voxel.grad[0] = grad_n0[i];
        voxel.grad[1] = grad_n1[i];
        voxel.grad[2] = grad_n2[i];

        tsdf_[idx] = voxel;

        // For vis_ - just as an example
        // Ensure counter_ is defined and accessible
        std::vector<bool> &vis_vec = vis_[idx];
        vis_vec.resize(counter_); 
        vis_vec.push_back(true);
    }

    return true;
}

