#pragma once
#include <algorithm>
#include <cctype>
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
    // PLYファイルを読み込む
    static void readPLY(std::ifstream& file, PointCloudCPU& points) {
        std::string line;

        // ヘッダー解析
        bool is_binary = false;
        int vertex_count = 0;
        std::vector<std::string> properties;
        std::vector<std::string> property_types;  // プロパティの型を格納
        int x_index = -1, y_index = -1, z_index = -1;

        // ヘッダー読み込み
        while (std::getline(file, line)) {
            if (line == "end_header") break;

            if (line.find("format binary") != std::string::npos) {
                is_binary = true;
            } else if (line.find("element vertex") != std::string::npos) {
                // "element vertex数字" の形式から数字部分を抽出
                std::string count_str = line.substr(line.find("vertex") + 6);
                vertex_count = std::stoi(count_str);
            } else if (line.find("property") != std::string::npos) {
                // プロパティ行を解析
                std::istringstream iss(line);
                std::string token1, token2, token3;
                iss >> token1 >> token2;

                // プロパティ名が最後のトークン
                if (token2 == "list") {
                    // リストプロパティの場合は形式が異なる
                    std::string token4, token5;
                    iss >> token3 >> token4 >> token5;
                    properties.push_back(token5);
                    property_types.push_back("list");
                } else {
                    iss >> token3;
                    properties.push_back(token3);
                    property_types.push_back(token2);  // token2はデータ型
                }

                if (token3 == "x") x_index = properties.size() - 1;
                if (token3 == "y") y_index = properties.size() - 1;
                if (token3 == "z") z_index = properties.size() - 1;
            }
        }

        // 頂点データが見つからない場合
        if (vertex_count <= 0 || x_index == -1 || y_index == -1 || z_index == -1) {
            throw std::runtime_error("Invalid PLY format: missing vertex data");
        }

        // バイナリ形式の場合
        if (is_binary) {
            // 各プロパティのサイズとオフセットを計算
            std::vector<int> property_sizes;
            std::vector<int> property_offsets;
            int total_row_size = 0;

            for (size_t i = 0; i < property_types.size(); ++i) {
                const std::string& type = property_types[i];
                int size = 0;

                // PLY形式の型サイズを決定
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
                    // リスト型は可変長なので、このシンプルな実装ではスキップ
                    // リストを含むデータを正確に処理するにはより複雑なロジックが必要
                    size = 0;
                }

                property_sizes.push_back(size);
                property_offsets.push_back(total_row_size);
                total_row_size += size;
            }

            // 行バッファを準備
            std::vector<char> buffer(total_row_size);

            for (int i = 0; i < vertex_count; i++) {
                file.read(buffer.data(), total_row_size);

                if (file.fail()) {
                    throw std::runtime_error("Error reading binary PLY data");
                }

                // x, y, z座標を抽出
                T x = 0, y = 0, z = 0;

                if (x_index >= 0) {
                    const int offset = property_offsets[x_index];
                    const std::string& type = property_types[x_index];

                    switch (type[0]) {
                        case 'f':  // float, float32
                            x = static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'd':  // double, float64
                            x = static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'i':  // int, int32, int16, int8
                            if (type == "int" || type == "int32") {
                                x = static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            } else if (type == "int16" || type == "short") {
                                x = static_cast<T>(*reinterpret_cast<int16_t*>(&buffer[offset]));
                            } else if (type == "int8" || type == "char") {
                                x = static_cast<T>(*reinterpret_cast<int8_t*>(&buffer[offset]));
                            }
                            break;
                        case 'u':  // uint, uint32, uint16, uint8, ushort, uchar
                            if (type == "uint" || type == "uint32") {
                                x = static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            } else if (type == "uint16" || type == "ushort") {
                                x = static_cast<T>(*reinterpret_cast<uint16_t*>(&buffer[offset]));
                            } else if (type == "uint8" || type == "uchar") {
                                x = static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[offset]));
                            }
                            break;
                    }
                }

                if (y_index >= 0) {
                    const int offset = property_offsets[y_index];
                    const std::string& type = property_types[y_index];

                    switch (type[0]) {
                        case 'f':  // float, float32
                            y = static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'd':  // double, float64
                            y = static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'i':  // int, int32, int16, int8
                            if (type == "int" || type == "int32") {
                                y = static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            } else if (type == "int16" || type == "short") {
                                y = static_cast<T>(*reinterpret_cast<int16_t*>(&buffer[offset]));
                            } else if (type == "int8" || type == "char") {
                                y = static_cast<T>(*reinterpret_cast<int8_t*>(&buffer[offset]));
                            }
                            break;
                        case 'u':  // uint, uint32, uint16, uint8, ushort, uchar
                            if (type == "uint" || type == "uint32") {
                                y = static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            } else if (type == "uint16" || type == "ushort") {
                                y = static_cast<T>(*reinterpret_cast<uint16_t*>(&buffer[offset]));
                            } else if (type == "uint8" || type == "uchar") {
                                y = static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[offset]));
                            }
                            break;
                    }
                }

                if (z_index >= 0) {
                    const int offset = property_offsets[z_index];
                    const std::string& type = property_types[z_index];

                    switch (type[0]) {
                        case 'f':  // float, float32
                            z = static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'd':  // double, float64
                            z = static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'i':  // int, int32, int16, int8
                            if (type == "int" || type == "int32") {
                                z = static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            } else if (type == "int16" || type == "short") {
                                z = static_cast<T>(*reinterpret_cast<int16_t*>(&buffer[offset]));
                            } else if (type == "int8" || type == "char") {
                                z = static_cast<T>(*reinterpret_cast<int8_t*>(&buffer[offset]));
                            }
                            break;
                        case 'u':  // uint, uint32, uint16, uint8, ushort, uchar
                            if (type == "uint" || type == "uint32") {
                                z = static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            } else if (type == "uint16" || type == "ushort") {
                                z = static_cast<T>(*reinterpret_cast<uint16_t*>(&buffer[offset]));
                            } else if (type == "uint8" || type == "uchar") {
                                z = static_cast<T>(*reinterpret_cast<uint8_t*>(&buffer[offset]));
                            }
                            break;
                    }
                }

                // 点を追加（if文の外に移動）
                points.points->emplace_back(x, y, z, static_cast<T>(1.0));
            }
        } else {
            // ASCII形式の場合
            for (int i = 0; i < vertex_count; i++) {
                std::getline(file, line);
                std::istringstream iss(line);

                std::vector<double> values(properties.size());
                for (size_t j = 0; j < properties.size(); j++) {
                    iss >> values[j];
                }

                T x = static_cast<T>(values[x_index]);
                T y = static_cast<T>(values[y_index]);
                T z = static_cast<T>(values[z_index]);

                points.points->emplace_back(x, y, z, static_cast<T>(1.0));
            }
        }
    }

    // PCDファイルを読み込む
    static void readPCD(std::ifstream& file, PointCloudCPU& points) {
        std::string line;

        // ヘッダー解析
        bool is_binary = false;
        int point_count = 0;
        int fields_count = 0;
        std::vector<std::string> fields;
        std::vector<std::string> field_types;  // フィールドの型を格納
        int x_index = -1, y_index = -1, z_index = -1;
        std::string data_type;

        while (std::getline(file, line)) {
            // コメント行をスキップ
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
                }
                fields_count = fields.size();
            } else if (keyword == "TYPE") {
                std::string type;
                while (iss >> type) {
                    field_types.push_back(type);
                }
            } else if (keyword == "POINTS") {
                iss >> point_count;
            } else if (keyword == "DATA") {
                iss >> data_type;
                if (data_type == "binary") {
                    is_binary = true;
                }
                break;  // DATAはヘッダーの最後
            }
        }

        // 点群データが見つからない場合
        if (point_count <= 0 || fields_count <= 0 || x_index == -1 || y_index == -1 || z_index == -1) {
            throw std::runtime_error("Invalid PCD format: missing point data");
        }

        // バイナリ形式の場合
        if (is_binary) {
            // 各フィールドのサイズとオフセットを計算
            std::vector<int> field_sizes;
            std::vector<int> field_offsets;
            int total_row_size = 0;

            // フィールド型情報がない場合はデフォルトでfloatを使用
            if (field_types.empty()) {
                field_types.resize(fields_count, "F");  // PCDではF=float32
            }

            for (int i = 0; i < fields_count; ++i) {
                const std::string& type = i < field_types.size() ? field_types[i] : "F";
                int size = 0;

                // PCD形式の型サイズを決定
                if (type == "I") {
                    size = 4;  // int32
                } else if (type == "U") {
                    size = 4;  // uint32
                } else if (type == "F") {
                    size = 4;  // float32
                } else if (type == "D") {
                    size = 8;  // float64/double
                }

                field_sizes.push_back(size);
                field_offsets.push_back(total_row_size);
                total_row_size += size;
            }

            // 行バッファを準備
            std::vector<char> buffer(total_row_size);

            for (int i = 0; i < point_count; i++) {
                file.read(buffer.data(), total_row_size);

                if (file.fail()) {
                    throw std::runtime_error("Error reading binary PCD data");
                }

                // x, y, z座標を抽出
                T x = 0, y = 0, z = 0;

                if (x_index >= 0 && x_index < field_types.size()) {
                    const int offset = field_offsets[x_index];
                    const std::string& type = field_types[x_index];

                    switch (type[0]) {
                        case 'F':  // float32
                            x = static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'D':  // double/float64
                            x = static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'I':  // int32
                            x = static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            break;
                        case 'U':  // uint32
                            x = static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            break;
                    }
                }

                if (y_index >= 0 && y_index < field_types.size()) {
                    const int offset = field_offsets[y_index];
                    const std::string& type = field_types[y_index];

                    switch (type[0]) {
                        case 'F':  // float32
                            y = static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'D':  // double/float64
                            y = static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'I':  // int32
                            y = static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            break;
                        case 'U':  // uint32
                            y = static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            break;
                    }
                }

                if (z_index >= 0 && z_index < field_types.size()) {
                    const int offset = field_offsets[z_index];
                    const std::string& type = field_types[z_index];

                    switch (type[0]) {
                        case 'F':  // float32
                            z = static_cast<T>(*reinterpret_cast<float*>(&buffer[offset]));
                            break;
                        case 'D':  // double/float64
                            z = static_cast<T>(*reinterpret_cast<double*>(&buffer[offset]));
                            break;
                        case 'I':  // int32
                            z = static_cast<T>(*reinterpret_cast<int32_t*>(&buffer[offset]));
                            break;
                        case 'U':  // uint32
                            z = static_cast<T>(*reinterpret_cast<uint32_t*>(&buffer[offset]));
                            break;
                    }
                }

                // 点を追加（if文の外に移動）
                points.points->emplace_back(x, y, z, static_cast<T>(1.0));
            }
        } else {
            // ASCII形式の場合
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
            }
        }
    }

public:
    static PointCloudCPU readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open PLY file: " + filename);
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
        }
        file.close();
        return cloud;
    }
};
}  // namespace sycl_points
