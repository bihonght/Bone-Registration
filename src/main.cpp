#include <iostream>
#include "mat.h"
#include "MeshLoader/meshSTLLoader.h"
#include "GradientSDF/Sdf.h"
#include "GradientSDF/MapGradPixelSdf.h"


void check_tsdf(const Sdf* tSDF);

// Main function
int main() {
        // Default input sequence in folder
    // std::string input = "";
    // std::string output = "../results/";
    // std::string stype = "map-gp";
    // std::string dtype = "";
    // size_t first = 0;
    // size_t last = 300;
    // float z_max = 3.5; // maximal depth to take into account in m
    // float sharp_threshold = 0.0001;
    // int num_frame = 30;

    float voxel_size = 0.1; // Voxel size in m
    float truncation_factor = 10; // truncation in voxels

    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> faces;
    std::vector<Eigen::Vector3f> vertex_normals; // Will store computed vertex normals

    std::string dir_cube1 = "/Users/apple/Documents/UTS master/Robotic Surgery/model/1inCube_Sync_wRounds_STL.stl";
    std::string dir_cube2 = "/Users/apple/Documents/UTS master/Robotic Surgery/model/cube.stl";
    std::string cube4_dir = "/Users/apple/Documents/UTS master/Robotic Surgery/model/4 Cube Retraction Calibration - 1159886/files/retraction_cube_test.stl";
    std::string sphere_dir = "/Users/apple/Documents/UTS master/Robotic Surgery/model/sphere - 4444345/files/sphere.stl";
    std::string bone_dir = "/Users/apple/Documents/UTS master/Robotic Surgery/model/Bone mesh model/1294-6leg_down.stl";

    if (MeshSTLLoader::load(bone_dir, vertices, faces)) {
        std::cout << "Loaded " << vertices.size() << " vertices and " << faces.size() << " faces from STL file." << std::endl;
        // Print some vertices and faces for verification
        for (size_t i = 0; i < std::min<size_t>(5, vertices.size()); ++i) {
            std::cout << "Vertex " << i << ": [" 
                      << vertices[i][0] << ", " 
                      << vertices[i][1] << ", " 
                      << vertices[i][2] << "]" << std::endl;
        }

        for (size_t i = 0; i < std::min<size_t>(5, faces.size()); ++i) {
            std::cout << "Face " << i << ": [" 
                      << faces[i][0] << ", " 
                      << faces[i][1] << ", " 
                      << faces[i][2] << "]" << std::endl;
        }
    } else {
        std::cerr << "Failed to load STL file." << std::endl;
        return -1;
    }

    std::cout << "Extracting faces..." << std::endl;
    // 1. Compute face normals
    std::vector<Eigen::Vector3f> face_normals(faces.size());
    for (size_t i = 0; i < faces.size(); i++) {
        Eigen::Vector3f v0 = vertices[faces[i][0]];
        Eigen::Vector3f v1 = vertices[faces[i][1]];
        Eigen::Vector3f v2 = vertices[faces[i][2]];
        face_normals[i] = (v1 - v0).cross(v2 - v0).normalized();
    }
    // 2. Accumulate into vertex normals
    // vertex_normals.resize(vertices.size(), Eigen::Vector3f::Zero());
    // for (size_t i = 0; i < faces.size(); i++) {
    //     vertex_normals[faces[i][0]] += face_normals[i];
    //     vertex_normals[faces[i][1]] += face_normals[i];
    //     vertex_normals[faces[i][2]] += face_normals[i];
    // }
    // // 3. Normalize vertex normals
    // for (size_t i = 0; i < vertices.size(); i++) {
    //     vertex_normals[i].normalize();
    // }
    const float truncation = truncation_factor * voxel_size;

    Sdf* tSDF;
    tSDF = new MapGradPixelSdf(voxel_size, truncation);
    std::cout << "Build SDF ..." << std::endl;
    // tSDF->updateFromMesh(vertices, faces);
    tSDF->updateFromMesh1(vertices, faces, vertex_normals);

    // std::cout << "Finish SDF build..." << std::endl;
    check_tsdf(tSDF);
    // // std::string filename = "mesh_4cube.ply";
    bool txt_extract = tSDF->save_sdf("sdf_1bone");
    // bool pc_extract = tSDF->extract_pc("cloud_rectangle_1.90.ply");
    // // bool mesh_extract = tSDF->extract_mesh(filename);
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