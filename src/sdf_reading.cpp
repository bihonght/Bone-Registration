#include <iostream>
#include <stdexcept>

#include "mat.h"
#include "MeshLoader/meshSTLLoader.h"
#include "GradientSDF/Sdf.h"
#include "GradientSDF/MapGradPixelSdf.h"


void check_tsdf(const Sdf* tSDF);

// Main function
int main() {

    float voxel_size = 0.05; // Voxel size in m
    float truncation_factor = 5; // truncation in voxels

    std::string filename = "sdf_1cube"; 
    std::string folder = filename + "_data";

    // Open grid info file from folder
    std::ifstream grid_file((folder + "/grid_info.txt").c_str());
    if (!grid_file.is_open()) {
        std::cerr << "Couldn't open grid_info file for loading!" << std::endl;
        throw std::invalid_argument( "received negative value" );
    }

    {
        std::string line;
        if (!std::getline(grid_file, line)) throw std::invalid_argument( "received negative value" );
        {
            std::istringstream iss(line);
            std::string dummy;
             // voxel size: <float>
            iss >> dummy >> dummy;
            iss >> voxel_size;
        }
    }

    const float truncation = truncation_factor * voxel_size;

    Sdf* tSDF;
    tSDF = new MapGradPixelSdf(voxel_size, truncation);
    std::cout << "Build SDF ..." << std::endl;
    bool load = tSDF->load_sdf(filename);
    // std::cout << "Finish SDF build..." << std::endl;
    // bool mesh_extract = tSDF->extract_mesh("mesh1cube.stl");

    check_tsdf(tSDF);
    // // std::string filename = "mesh_4cube.ply";
    delete tSDF;

    return 0;
}

void check_tsdf(const Sdf* tSDF){
    struct Point3D {
                    double x;
                    double y;
                    double z;
                    };

    std::vector<Point3D> points = {
        {0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2},
        {0.25, 0.25, 0.25},
        {0.35, 0.35, -0.35},
        {-0.4, 0, 0.1},
        {0.55, 0.55, 0.55},
        {0.55, 0.35, -0.35},
        // Additional points can be easily added here
        // {1.1, 1.1, 1.1},
        // {0.9, 0.9, 0.9},
        // {1.1, 0, 0.8},
        // {1, 1.1, 1.1},
        // {0, 0, 0.9}
    };

    for (int i = 0; i < points.size(); ++i) {
        Eigen::Vector3f grad;  
        Eigen::Vector3f point = Eigen::Vector3f(points[i].x, points[i].y, points[i].z); 
        float signed_dist = tSDF->tsdf(Eigen::Vector3f(points[i].x, points[i].y, points[i].z), &grad);
        std::cout << "Point " << i << " is " << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
        std::cout << "Point " << i << ": signed distance = " << signed_dist
                  << ", grad = [" << grad[0] << ", " << grad[1] << ", " << grad[2] << "]" << std::endl;
    }
}