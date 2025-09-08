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
        const bool has_rgb = cloud.has_rgb();

        // Count valid points first
        const size_t valid_count = countValidPoints(cloud);

        if (valid_count == 0) {
            throw std::runtime_error("No valid points to write");
        }

        // Set memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.set_accessed_by_host(cloud.points_ptr(), N);
            if (has_rgb) {
                cloud.queue.set_accessed_by_host(cloud.rgb_ptr(), N);
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

        if (has_rgb) {
            file << "property uchar red\n";
            file << "property uchar green\n";
            file << "property uchar blue\n";
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

                if (has_rgb) {
                    const auto& c = (*cloud.rgb)[i];
                    const uint8_t rgb[3] = {static_cast<uint8_t>(std::clamp(c.x(), 0.f, 1.f) * 255.f),
                                            static_cast<uint8_t>(std::clamp(c.y(), 0.f, 1.f) * 255.f),
                                            static_cast<uint8_t>(std::clamp(c.z(), 0.f, 1.f) * 255.f)};
                    file.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
                }
            }
        } else {
            // ASCII format
            file << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < N; ++i) {
                const auto& point = (*cloud.points)[i];
                if (!isValidPoint(point)) continue;

                file << point.x() << " " << point.y() << " " << point.z();

                if (has_rgb) {
                    const auto& c = (*cloud.rgb)[i];
                    file << " "  //
                         << static_cast<int>(std::clamp(c.x(), 0.f, 1.f) * 255.f) << " "
                         << static_cast<int>(std::clamp(c.y(), 0.f, 1.f) * 255.f) << " "
                         << static_cast<int>(std::clamp(c.z(), 0.f, 1.f) * 255.f);
                }
                file << "\n";
            }
        }

        // Clear memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.clear_accessed_by_host(cloud.points_ptr(), N);
            if (has_rgb) {
                cloud.queue.clear_accessed_by_host(cloud.rgb_ptr(), N);
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
        const bool has_rgb = cloud.has_rgb();

        // Count valid points first
        const size_t valid_count = countValidPoints(cloud);

        if (valid_count == 0) {
            throw std::runtime_error("No valid points to write");
        }

        // Set memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.set_accessed_by_host(cloud.points_ptr(), N);
            if (has_rgb) {
                cloud.queue.set_accessed_by_host(cloud.rgb_ptr(), N);
            }
        }

        // Write PCD header
        file << "# .PCD v.7 - Point Cloud Data file format\n";
        file << "VERSION .7\n";

        file << "FIELDS x y z";
        if (has_rgb) {
            file << " rgb";
        }
        file << "\n";

        file << "SIZE 4 4 4";
        if (has_rgb) {
            file << " 4";
        }
        file << "\n";

        file << "TYPE F F F";
        if (has_rgb) {
            file << " U";
        }
        file << "\n";

        file << "COUNT 1 1 1";
        if (has_rgb) {
            file << " 1";
        }
        file << "\n";

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

                if (has_rgb) {
                    const auto& color = (*cloud.rgb)[i];
                    const int32_t rgb = (static_cast<uint32_t>(color.x() * 255.0f) << 16) |
                                        (static_cast<uint32_t>(color.y() * 255.0f) << 8) |
                                        (static_cast<uint32_t>(color.z() * 255.0f));
                    file.write(reinterpret_cast<const char*>(&rgb), sizeof(rgb));
                }
            }
        } else {
            // ASCII format
            file << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < N; ++i) {
                const auto& point = (*cloud.points)[i];
                if (!isValidPoint(point)) continue;

                file << point.x() << " " << point.y() << " " << point.z();

                if (has_rgb) {
                    const auto& color = (*cloud.rgb)[i];
                    const int32_t rgb = (static_cast<uint32_t>(color.x() * 255.0f) << 16) |
                                        (static_cast<uint32_t>(color.y() * 255.0f) << 8) |
                                        (static_cast<uint32_t>(color.z() * 255.0f));
                    file << " " << rgb;
                }
                file << "\n";
            }
        }

        // Clear memory access hints for shared memory
        if constexpr (std::is_same_v<PointCloud, PointCloudShared>) {
            cloud.queue.clear_accessed_by_host(cloud.points_ptr(), N);
            if (has_rgb) {
                cloud.queue.clear_accessed_by_host(cloud.rgb_ptr(), N);
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
