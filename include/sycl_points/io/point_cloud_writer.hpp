#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

class PointCloudWriter {
private:
    // Check if point has valid coordinates
    static bool isValidPoint(const PointType& point) {
        return std::isfinite(point.x()) && std::isfinite(point.y()) && std::isfinite(point.z());
    }

    // Get file extension (lowercase)
    static std::string get_file_extension(const std::string& filename) {
        std::string extension;
        size_t dot_pos = filename.find_last_of(".");
        if (dot_pos != std::string::npos) {
            extension = filename.substr(dot_pos + 1);
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
        }
        return extension;
    }

    // Count valid points (non-NaN/Inf)
    template <typename PointCloud>
    static size_t countValidPoints(const PointCloud& cloud) {
        size_t valid_count = 0;
        const size_t N = cloud.size();

        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            // For shared memory, set access hint to host
            cloud.queue.set_accessed_by_host(cloud.points_ptr(), N);
        }

        for (size_t i = 0; i < N; ++i) {
            if (isValidPoint((*cloud.points)[i])) {
                ++valid_count;
            }
        }

        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            // Clear access hints
            cloud.queue.clear_accessed_by_host(cloud.points_ptr(), N);
        }

        return valid_count;
    }

    // Write PLY format
    template <typename PointCloud>
    static void writePLY(std::ofstream& file, const PointCloud& cloud, bool binary) {
        const size_t N = cloud.size();
        const bool has_normals = cloud.has_normal();

        // Count valid points first
        const size_t valid_count = countValidPoints(cloud);

        if (valid_count == 0) {
            throw std::runtime_error("No valid points to write");
        }

        // Set memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.set_accessed_by_host(cloud.points_ptr(), N);
            if (has_normals) {
                cloud.queue.set_accessed_by_host(cloud.normals_ptr(), N);
            }
        }

        // Write PLY header
        file << "ply\n";
        if (binary) {
            file << "format binary_little_endian 1.0\n";
        } else {
            file << "format ascii 1.0\n";
        }
        file << "element vertex " << valid_count << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";

        if (has_normals) {
            file << "property float nx\n";
            file << "property float ny\n";
            file << "property float nz\n";
        }

        file << "end_header\n";

        if (file.fail()) {
            throw std::runtime_error("Failed to write PLY header");
        }

        // Write data
        if (binary) {
            // Binary format
            for (size_t i = 0; i < N; ++i) {
                const auto& point = (*cloud.points)[i];
                if (!isValidPoint(point)) continue;

                float coords[3] = {point.x(), point.y(), point.z()};
                file.write(reinterpret_cast<const char*>(coords), sizeof(coords));

                if (has_normals) {
                    const auto& normal = (*cloud.normals)[i];
                    float normals[3] = {normal.x(), normal.y(), normal.z()};
                    file.write(reinterpret_cast<const char*>(normals), sizeof(normals));
                }
            }
        } else {
            // ASCII format
            file << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < N; ++i) {
                const auto& point = (*cloud.points)[i];
                if (!isValidPoint(point)) continue;

                file << point.x() << " " << point.y() << " " << point.z();

                if (has_normals) {
                    const auto& normal = (*cloud.normals)[i];
                    file << " " << normal.x() << " " << normal.y() << " " << normal.z();
                }
                file << "\n";
            }
        }

        // Clear memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.clear_accessed_by_host(cloud.points_ptr(), N);
            if (has_normals) {
                cloud.queue.clear_accessed_by_host(cloud.normals_ptr(), N);
            }
        }

        if (file.fail()) {
            throw std::runtime_error("Failed to write PLY data");
        }
    }

    // Write PCD format
    template <typename PointCloud>
    static void writePCD(std::ofstream& file, const PointCloud& cloud, bool binary) {
        const size_t N = cloud.size();
        const bool has_normals = cloud.has_normal();

        // Count valid points first
        const size_t valid_count = countValidPoints(cloud);

        if (valid_count == 0) {
            throw std::runtime_error("No valid points to write");
        }

        // Set memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.set_accessed_by_host(cloud.points_ptr(), N);
            if (has_normals) {
                cloud.queue.set_accessed_by_host(cloud.normals_ptr(), N);
            }
        }

        // Write PCD header
        file << "# .PCD v0.7 - Point Cloud Data file format\n";
        file << "VERSION 0.7\n";

        if (has_normals) {
            file << "FIELDS x y z nx ny nz\n";
            file << "SIZE 4 4 4 4 4 4\n";
            file << "TYPE F F F F F F\n";
            file << "COUNT 1 1 1 1 1 1\n";
        } else {
            file << "FIELDS x y z\n";
            file << "SIZE 4 4 4\n";
            file << "TYPE F F F\n";
            file << "COUNT 1 1 1\n";
        }

        file << "WIDTH " << valid_count << "\n";
        file << "HEIGHT 1\n";
        file << "VIEWPOINT 0 0 0 1 0 0 0\n";
        file << "POINTS " << valid_count << "\n";

        if (binary) {
            file << "DATA binary\n";
        } else {
            file << "DATA ascii\n";
        }

        if (file.fail()) {
            throw std::runtime_error("Failed to write PCD header");
        }

        // Write data
        if (binary) {
            // Binary format
            for (size_t i = 0; i < N; ++i) {
                const auto& point = (*cloud.points)[i];
                if (!isValidPoint(point)) continue;

                float coords[3] = {point.x(), point.y(), point.z()};
                file.write(reinterpret_cast<const char*>(coords), sizeof(coords));

                if (has_normals) {
                    const auto& normal = (*cloud.normals)[i];
                    float normals[3] = {normal.x(), normal.y(), normal.z()};
                    file.write(reinterpret_cast<const char*>(normals), sizeof(normals));
                }
            }
        } else {
            // ASCII format
            file << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < N; ++i) {
                const auto& point = (*cloud.points)[i];
                if (!isValidPoint(point)) continue;

                file << point.x() << " " << point.y() << " " << point.z();

                if (has_normals) {
                    const auto& normal = (*cloud.normals)[i];
                    file << " " << normal.x() << " " << normal.y() << " " << normal.z();
                }
                file << "\n";
            }
        }

        // Clear memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.clear_accessed_by_host(cloud.points_ptr(), N);
            if (has_normals) {
                cloud.queue.clear_accessed_by_host(cloud.normals_ptr(), N);
            }
        }

        if (file.fail()) {
            throw std::runtime_error("Failed to write PCD data");
        }
    }

public:
    /// @brief Write point cloud to file
    /// @param filename Output file path
    /// @param cloud Point cloud data in CPU memory
    /// @param binary Whether to use binary format (default: false = ASCII)
    static void writeFile(const std::string& filename, const PointCloudCPU& cloud, bool binary = false) {
        if (cloud.size() == 0) {
            throw std::runtime_error("Cannot write empty point cloud");
        }

        std::ofstream file;
        if (binary) {
            file.open(filename, std::ios::out | std::ios::binary);
        } else {
            file.open(filename, std::ios::out);
        }

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        const std::string extension = get_file_extension(filename);

        try {
            if (extension == "ply") {
                writePLY(file, cloud, binary);
            } else if (extension == "pcd") {
                writePCD(file, cloud, binary);
            } else {
                throw std::runtime_error("Unsupported file format: " + extension);
            }
        } catch (const std::exception& e) {
            file.close();
            throw;
        }

        file.close();
        if (file.fail()) {
            throw std::runtime_error("Failed to close file: " + filename);
        }
    }

    /// @brief Write point cloud to file
    /// @param filename Output file path
    /// @param cloud Point cloud data in shared memory
    /// @param binary Whether to use binary format (default: false = ASCII)
    static void writeFile(const std::string& filename, const PointCloudShared& cloud, bool binary = false) {
        if (cloud.size() == 0) {
            throw std::runtime_error("Cannot write empty point cloud");
        }

        std::ofstream file;
        if (binary) {
            file.open(filename, std::ios::out | std::ios::binary);
        } else {
            file.open(filename, std::ios::out);
        }

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        const std::string extension = get_file_extension(filename);

        try {
            if (extension == "ply") {
                writePLY(file, cloud, binary);
            } else if (extension == "pcd") {
                writePCD(file, cloud, binary);
            } else {
                throw std::runtime_error("Unsupported file format: " + extension);
            }
        } catch (const std::exception& e) {
            file.close();
            throw;
        }

        file.close();
        if (file.fail()) {
            throw std::runtime_error("Failed to close file: " + filename);
        }
    }

    /// @brief Write PLY format file
    /// @param filename Output file path
    /// @param cloud Point cloud data in CPU memory
    /// @param binary Whether to use binary format (default: false = ASCII)
    static void writePLY(const std::string& filename, const PointCloudCPU& cloud, bool binary = false) {
        writeFile(filename + (filename.find(".ply") == std::string::npos ? ".ply" : ""), cloud, binary);
    }

    /// @brief Write PLY format file
    /// @param filename Output file path
    /// @param cloud Point cloud data in shared memory
    /// @param binary Whether to use binary format (default: false = ASCII)
    static void writePLY(const std::string& filename, const PointCloudShared& cloud, bool binary = false) {
        writeFile(filename + (filename.find(".ply") == std::string::npos ? ".ply" : ""), cloud, binary);
    }

    /// @brief Write PCD format file
    /// @param filename Output file path
    /// @param cloud Point cloud data in CPU memory
    /// @param binary Whether to use binary format (default: false = ASCII)
    static void writePCD(const std::string& filename, const PointCloudCPU& cloud, bool binary = false) {
        writeFile(filename + (filename.find(".pcd") == std::string::npos ? ".pcd" : ""), cloud, binary);
    }

    /// @brief Write PCD format file
    /// @param filename Output file path
    /// @param cloud Point cloud data in shared memory
    /// @param binary Whether to use binary format (default: false = ASCII)
    static void writePCD(const std::string& filename, const PointCloudShared& cloud, bool binary = false) {
        writeFile(filename + (filename.find(".pcd") == std::string::npos ? ".pcd" : ""), cloud, binary);
    }
};

}  // namespace sycl_points
