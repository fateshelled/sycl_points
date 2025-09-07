#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sycl_points/points/point_cloud.hpp>
#include <vector>

namespace sycl_points {

class PointCloudReader {
private:
    using T = float;
    // Read PLY file
    static void readPLY(std::ifstream& file, PointCloudCPU& points) {
        std::string line;

        // Header parsing
        bool is_binary = false;
        int vertex_count = 0;
        std::vector<std::string> properties;
        std::vector<std::string> property_types;  // Store property types
        int x_index = -1, y_index = -1, z_index = -1;
        int rgb_r_index = -1, rgb_g_index = -1, rgb_b_index = -1;

        // Read header
        while (std::getline(file, line)) {
            if (line == "end_header") break;

            if (line.find("format binary") != std::string::npos) {
                is_binary = true;
            } else if (line.find("element vertex") != std::string::npos) {
                // Extract number part from "element vertex<number>" format
                std::string count_str = line.substr(line.find("vertex") + 6);
                vertex_count = std::stoi(count_str);
            } else if (line.find("property") != std::string::npos) {
                // Parse property line
                std::istringstream iss(line);
                std::string token1, token2, token3;
                iss >> token1 >> token2;

                // Property name is the last token
                if (token2 == "list") {
                    // // List properties have different format
                    // std::string token4, token5;
                    // iss >> token3 >> token4 >> token5;
                    // properties.push_back(token5);
                    // property_types.push_back("list");
                } else {
                    iss >> token3;
                    properties.push_back(token3);
                    property_types.push_back(token2);  // token2 is data type
                }

                if (token3 == "x") x_index = properties.size() - 1;
                if (token3 == "y") y_index = properties.size() - 1;
                if (token3 == "z") z_index = properties.size() - 1;
                if (token3 == "red") rgb_r_index = properties.size() - 1;
                if (token3 == "green") rgb_g_index = properties.size() - 1;
                if (token3 == "blue") rgb_b_index = properties.size() - 1;
            }
        }

        // When vertex data is not found
        if (vertex_count <= 0 || x_index == -1 || y_index == -1 || z_index == -1) {
            throw std::runtime_error("Invalid PLY format: missing vertex data");
        }

        points.points->clear();
        points.points->reserve(vertex_count);

        const bool has_rgb = (rgb_r_index >= 0 && rgb_g_index >= 0 && rgb_b_index >= 0);
        points.rgb->clear();
        if (has_rgb) {
            points.rgb->reserve(vertex_count);
        }

        // For binary format
        if (is_binary) {
            // Calculate size and offset for each property
            std::vector<int> property_sizes;
            std::vector<int> property_offsets;
            int total_row_size = 0;

            for (size_t i = 0; i < property_types.size(); ++i) {
                const std::string& type = property_types[i];
                int size = 0;

                // Determine PLY format type sizes
                if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") {
                    size = 1;
                } else if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") {
                    size = 2;
                } else if (type == "int" || type == "uint" || type == "int32" || type == "uint32" || type == "float" ||
                           type == "float32") {
                    size = 4;
                } else if (type == "double" || type == "float64") {
                    size = 8;
                } else if (type == "list") {
                    // List type is variable length, skip in this simple implementation
                    // More complex logic needed to accurately process list-containing data
                    size = 0;
                }

                property_sizes.push_back(size);
                property_offsets.push_back(total_row_size);
                total_row_size += size;
            }

            // Prepare row buffer
            std::vector<char> buffer(total_row_size);

            for (int i = 0; i < vertex_count; i++) {
                file.read(buffer.data(), total_row_size);

                if (file.fail()) {
                    throw std::runtime_error("Error reading binary PLY data");
                }

                // Extract x, y, z coordinates
                auto read_coordinate = [&](int idx) {
                    const int offset = property_offsets[idx];
                    const std::string& type = property_types[idx];
                    switch (type[0]) {
                        case 'f':  // float, float32
                            return static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'd':  // double, float64
                            return static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'i':  // int, int32, int16, int8
                            if (type == "int" || type == "int32") {
                                return static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            } else if (type == "int16" || type == "short") {
                                return static_cast<T>(*reinterpret_cast<int16_t*>(&buffer[offset]));
                            } else if (type == "int8" || type == "char") {
                                return static_cast<T>(*reinterpret_cast<int8_t*>(&buffer[offset]));
                            }
                            break;
                        case 'u':  // uint, uint32, uint16, uint8, ushort, uchar
                            if (type == "uint" || type == "uint32") {
                                return static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            } else if (type == "uint16" || type == "ushort") {
                               return static_cast<T>(*reinterpret_cast<uint16_t*>(&buffer[offset]));
                            } else if (type == "uint8" || type == "uchar") {
                                return static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[offset]));
                            }
                            break;
                    }
                    static_assert("unsupported type");
                    return (T)0;
                };

                const T x = read_coordinate(x_index);
                const T y = read_coordinate(y_index);
                const T z = read_coordinate(z_index);
                points.points->emplace_back(x, y, z, static_cast<T>(1.0));

                if (has_rgb) {
                    auto read_color = [&](int idx) {
                        const int offset = property_offsets[idx];
                        const std::string& type = property_types[idx];
                        switch (type[0]) {
                            case 'i':
                                if (type == "int" || type == "int32") {
                                    return static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                                } else if (type == "int16" || type == "short") {
                                    return static_cast<T>(*reinterpret_cast<int16_t*>(&buffer[offset]));
                                } else {
                                    return static_cast<T>(*reinterpret_cast<int8_t*>(&buffer[offset]));
                                }
                            case 'u':
                                if (type == "uint" || type == "uint32") {
                                    return static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                                } else if (type == "uint16" || type == "ushort") {
                                    return static_cast<T>(*reinterpret_cast<uint16_t*>(&buffer[offset]));
                                } else {
                                    return static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[offset]));
                                }
                            default:
                                static_assert("unsupported type");
                        }
                        return (T)0;
                    };
                    const T r = read_color(rgb_r_index);
                    const T g = read_color(rgb_g_index);
                    const T b = read_color(rgb_b_index);
                    points.rgb->emplace_back(r / 255.f, g / 255.f, b / 255.f, static_cast<T>(1.0));
                }
            }
        } else {
            // For ASCII format
            std::vector<double> raw_values(properties.size());
            for (int i = 0; i < vertex_count; i++) {
                size_t count = 0;
                while (count < properties.size()) {
                    std::getline(file, line);
                    std::istringstream iss(line);
                    double raw_value = std::numeric_limits<double>::quiet_NaN();
                    while (iss >> raw_value) {
                        if (!std::isnan(raw_value)) {
                            raw_values[count++] = raw_value;
                        } else {
                            break;
                        }
                        raw_value = std::numeric_limits<double>::quiet_NaN();
                    }
                }

                const T x = static_cast<T>(raw_values[x_index]);
                const T y = static_cast<T>(raw_values[y_index]);
                const T z = static_cast<T>(raw_values[z_index]);
                points.points->emplace_back(x, y, z, static_cast<T>(1.0));

                if (has_rgb) {
                    const T r = static_cast<T>(raw_values[rgb_r_index]) / 255.f;
                    const T g = static_cast<T>(raw_values[rgb_g_index]) / 255.f;
                    const T b = static_cast<T>(raw_values[rgb_b_index]) / 255.f;
                    points.rgb->emplace_back(r, g, b, static_cast<T>(1.0));
                }
            }
        }
    }

    // Read PCD file
    static void readPCD(std::ifstream& file, PointCloudCPU& points) {
        std::string line;

        // Header parsing
        bool is_binary = false;
        int point_count = 0;
        int fields_count = 0;
        std::vector<std::string> fields;
        std::vector<std::string> field_types;  // Store field types
        std::vector<int> field_sizes_line;     // Store sizes from SIZE line
        int x_index = -1, y_index = -1, z_index = -1;
        int rgb_r_index = -1, rgb_g_index = -1, rgb_b_index = -1;
        std::string data_type;

        while (std::getline(file, line)) {
            // Skip comment lines
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "FIELDS") {
                std::string field;
                while (iss >> field) {
                    fields.push_back(field);
                    if (field == "x") x_index = fields.size() - 1;
                    if (field == "y") y_index = fields.size() - 1;
                    if (field == "z") z_index = fields.size() - 1;
                    if (field == "r") rgb_r_index = fields.size() - 1;
                    if (field == "g") rgb_g_index = fields.size() - 1;
                    if (field == "b") rgb_b_index = fields.size() - 1;
                }
                fields_count = fields.size();
            } else if (keyword == "TYPE") {
                std::string type;
                while (iss >> type) {
                    field_types.push_back(type);
                }
            } else if (keyword == "SIZE") {
                int size;
                while (iss >> size) {
                    field_sizes_line.push_back(size);
                }
            } else if (keyword == "POINTS") {
                iss >> point_count;
            } else if (keyword == "DATA") {
                iss >> data_type;
                if (data_type == "binary") {
                    is_binary = true;
                }
                break;  // DATA is the last header line
            }
        }

        // When point cloud data is not found
        if (point_count <= 0 || fields_count <= 0 || x_index == -1 || y_index == -1 || z_index == -1) {
            throw std::runtime_error("Invalid PCD format: missing point data");
        }

        points.points->clear();
        points.points->reserve(point_count);
        points.rgb->clear();

        bool has_rgb = false;
        if (rgb_r_index >= 0 && rgb_g_index >= 0 && rgb_b_index >= 0) {
            if (field_sizes_line.size() > static_cast<size_t>(std::max({rgb_r_index, rgb_g_index, rgb_b_index})) &&
                field_sizes_line[rgb_r_index] == 1 && field_sizes_line[rgb_g_index] == 1 && field_sizes_line[rgb_b_index] == 1) {
                has_rgb = true;
                points.rgb->reserve(point_count);
            }
        }

        // For binary format
        if (is_binary) {
            // Calculate size and offset for each field
            std::vector<int> field_sizes;
            std::vector<int> field_offsets;
            int total_row_size = 0;

            // Use float as default when field type info is missing
            if (field_types.empty()) {
                field_types.resize(fields_count, "F");  // F=float32 in PCD
            }

            for (int i = 0; i < fields_count; ++i) {
                int size = i < field_sizes_line.size() ? field_sizes_line[i] : 0;
                if (size == 0) {
                    const std::string& type = i < field_types.size() ? field_types[i] : "F";
                    // Determine PCD format type sizes when SIZE is missing
                    if (type == "I" || type == "U" || type == "F") {
                        size = 4;  // int32/uint32/float32
                    } else if (type == "D") {
                        size = 8;  // float64/double
                    }
                }

                field_sizes.push_back(size);
                field_offsets.push_back(total_row_size);
                total_row_size += size;
            }

            // Prepare row buffer
            std::vector<char> buffer(total_row_size);

            for (int i = 0; i < point_count; i++) {
                file.read(buffer.data(), total_row_size);

                if (file.fail()) {
                    throw std::runtime_error("Error reading binary PCD data");
                }

                // Extract x, y, z coordinates
                auto read_coordinate = [&](int idx) {
                    const int offset = field_offsets[idx];
                    const std::string& type = field_types[idx];
                    switch (type[0]) {
                        case 'F':  // float32
                            return static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'D':  // double/float64
                            return static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'I':  // int32
                            return static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            break;
                        case 'U':  // uint32
                            return static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            break;
                    }
                    static_assert("unsupported type");
                    return (T)0;
                };

                const T x = read_coordinate(x_index);
                const T y = read_coordinate(y_index);
                const T z = read_coordinate(z_index);
                // Add point and color
                points.points->emplace_back(x, y, z, static_cast<T>(1.0));

                if (has_rgb) {
                    // RGB fieid is uint8_t
                    const T r = static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[field_offsets[rgb_r_index]]));
                    const T g = static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[field_offsets[rgb_g_index]]));
                    const T b = static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[field_offsets[rgb_b_index]]));
                    points.rgb->emplace_back(r / 255.f, g / 255.f, b / 255.f, static_cast<T>(1.0));
                }

            }
        } else {
            // For ASCII format
            for (int i = 0; i < point_count; i++) {
                std::getline(file, line);
                std::istringstream iss(line);

                std::vector<double> values(fields_count);
                for (int j = 0; j < fields_count; j++) {
                    iss >> values[j];
                }

                T x = static_cast<T>(values[x_index]);
                T y = static_cast<T>(values[y_index]);
                T z = static_cast<T>(values[z_index]);
                points.points->emplace_back(x, y, z, static_cast<T>(1.0));

                if (has_rgb) {
                    T r = static_cast<T>(values[rgb_r_index]) / 255.f;
                    T g = static_cast<T>(values[rgb_g_index]) / 255.f;
                    T b = static_cast<T>(values[rgb_b_index]) / 255.f;
                    points.rgb->emplace_back(r, g, b, static_cast<T>(1.0));
                }
            }
        }
    }

public:
    /// @brief Read point cloud file to CPU memory
    /// @param filename File path to read
    /// @return Point cloud data in CPU memory
    static PointCloudCPU readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Extract file extension from filename
        std::string extension;
        size_t dot_pos = filename.find_last_of(".");
        if (dot_pos != std::string::npos) {
            extension = filename.substr(dot_pos + 1);
            // Convert to lowercase for case-insensitive comparison
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
        }

        PointCloudCPU cloud;
        if (extension == "ply") {
            std::string first_line;
            std::getline(file, first_line);
            if (first_line != "ply") {
                throw std::runtime_error("Invalid PLY file format: " + filename);
            }
            readPLY(file, cloud);
        } else if (extension == "pcd") {
            readPCD(file, cloud);
        } else {
            std::cerr << "not supported format [" << extension << "]" << std::endl;
            throw std::runtime_error("Unsupported file format: " + extension);
        }
        file.close();
        return cloud;
    }

    /// @brief Read point cloud file to shared memory
    /// @param filename File path to read
    /// @param queue SYCL device queue for shared memory allocation
    /// @return Point cloud data in shared memory
    static PointCloudShared readFile(const std::string& filename, const sycl_utils::DeviceQueue& queue) {
        // First read to CPU memory
        PointCloudCPU cpu_cloud = readFile(filename);

        // Then convert to shared memory
        return PointCloudShared(queue, cpu_cloud);
    }

    /// @brief Read point cloud file to shared memory (convenience method)
    /// @param queue SYCL device queue for shared memory allocation
    /// @param filename File path to read
    /// @return Point cloud data in shared memory
    static PointCloudShared readFile(const sycl_utils::DeviceQueue& queue, const std::string& filename) {
        return readFile(filename, queue);
    }
};
}  // namespace sycl_points
