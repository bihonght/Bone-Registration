#ifndef MESH_STL_LOADER_H
#define MESH_STL_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>

// Hash function for Eigen::Vector3f
struct Vector3fHash {
    std::size_t operator()(const Eigen::Vector3f& v) const {
        std::hash<float> hasher;
        return hasher(v.x()) ^ (hasher(v.y()) << 1) ^ (hasher(v.z()) << 2);
    }
};

// Equality comparator for Eigen::Vector3f
struct Vector3fEqual {
    bool operator()(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) const {
        return v1.isApprox(v2, 1e-6); // Allows for a small tolerance in floating-point comparison
    }
};

class MeshSTLLoader {
public:
    // Function to load vertices and faces from an STL file
    static bool load(const std::string& filename, std::vector<Eigen::Vector3f>& vertices, std::vector<Eigen::Vector3i>& faces) {
        vertices.clear();
        faces.clear();

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open STL file: " << filename << std::endl;
            return false;
        }

        // STL header is 80 bytes
        char header[80];
        file.read(header, 80);

        // Number of triangles is a 4-byte unsigned int
        uint32_t num_triangles;
        file.read(reinterpret_cast<char*>(&num_triangles), sizeof(num_triangles));

        // Map to store unique vertices and their indices
        std::unordered_map<Eigen::Vector3f, size_t, Vector3fHash, Vector3fEqual> vertex_map;

        // Read each triangle
        for (uint32_t i = 0; i < num_triangles; ++i) {
            char buffer[50];
            file.read(buffer, 50); // Each triangle is 50 bytes

            // Ignore the normal vector
            Eigen::Vector3f v0(
                *reinterpret_cast<float*>(&buffer[12]),
                *reinterpret_cast<float*>(&buffer[16]),
                *reinterpret_cast<float*>(&buffer[20])
            );
            Eigen::Vector3f v1(
                *reinterpret_cast<float*>(&buffer[24]),
                *reinterpret_cast<float*>(&buffer[28]),
                *reinterpret_cast<float*>(&buffer[32])
            );
            Eigen::Vector3f v2(
                *reinterpret_cast<float*>(&buffer[36]),
                *reinterpret_cast<float*>(&buffer[40]),
                *reinterpret_cast<float*>(&buffer[44])
            );

            // Add unique vertices and retrieve indices
            size_t v0_idx = addVertex(v0, vertices, vertex_map);
            size_t v1_idx = addVertex(v1, vertices, vertex_map);
            size_t v2_idx = addVertex(v2, vertices, vertex_map);

            // Add a face using the indices of the vertices
            faces.emplace_back(v0_idx, v1_idx, v2_idx);
        }

        file.close();
        return true;
    }

private:
    // Helper function to add a vertex if it's unique
    static size_t addVertex(const Eigen::Vector3f& vertex, 
                            std::vector<Eigen::Vector3f>& vertices, 
                            std::unordered_map<Eigen::Vector3f, size_t, Vector3fHash, Vector3fEqual>& vertex_map) {
        auto it = vertex_map.find(vertex);
        if (it != vertex_map.end()) {
            return it->second; // Vertex already exists, return its index
        }

        size_t new_index = vertices.size();
        vertices.push_back(vertex);    // Add the vertex to the list
        vertex_map[vertex] = new_index; // Store the index in the map
        return new_index;
    }
};

#endif // MESH_STL_LOADER_H
