#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {

namespace ros2 {
inline PointCloudShared::Ptr fromROS2msg(const sycl_points::sycl_utils::DeviceQueue& queue,
                                         const sensor_msgs::msg::PointCloud2& cloud) {
    uint8_t x_type = 0;
    uint8_t y_type = 0;
    uint8_t z_type = 0;

    int32_t x_offset = -1;
    int32_t y_offset = -1;
    int32_t z_offset = -1;
    for (const auto& field : cloud.fields) {
        if (field.name == "x") {
            x_type = field.datatype;
            x_offset = field.offset;
        } else if (field.name == "y") {
            y_type = field.datatype;
            y_offset = field.offset;
        } else if (field.name == "z") {
            z_type = field.datatype;
            z_offset = field.offset;
        }
    }
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        std::cerr << "Not found point cloud field" << std::endl;
        return nullptr;
    }
    if ((x_type != sensor_msgs::msg::PointField::FLOAT32 && x_type != sensor_msgs::msg::PointField::FLOAT64)) {
        std::cerr << "Not supported point field type" << std::endl;
        return nullptr;
    }
    if (x_type != y_type || x_type != z_type) {
        std::cerr << "Not supported point field type" << std::endl;
        return nullptr;
    }

    PointCloudShared::Ptr ret = std::make_shared<PointCloudShared>(queue);
    const size_t point_size = cloud.width * cloud.height;
    ret->resize_points(point_size);
    const size_t point_step = cloud.point_step;
    if (x_type == sensor_msgs::msg::PointField::FLOAT32) {
        if (x_offset + sizeof(float) == y_offset && y_offset + sizeof(float) == z_offset) {
            for (size_t i = 0; i < point_size; ++i) {
                const auto x_ptr = reinterpret_cast<const float*>(&cloud.data[point_step * i + x_offset]);
                ret->points->data()[i] = PointType(x_ptr[0], x_ptr[1], x_ptr[2], 1.0f);
            }
        } else {
            for (size_t i = 0; i < point_size; ++i) {
                const auto x = reinterpret_cast<const float*>(&cloud.data[point_step * i + x_offset])[0];
                const auto y = reinterpret_cast<const float*>(&cloud.data[point_step * i + y_offset])[0];
                const auto z = reinterpret_cast<const float*>(&cloud.data[point_step * i + z_offset])[0];
                ret->points->data()[i] = PointType(x, y, z, 1.0f);
            }
        }
    } else if (x_type == sensor_msgs::msg::PointField::FLOAT64) {
        if (x_offset + sizeof(float) == y_offset && y_offset + sizeof(float) == z_offset) {
            for (size_t i = 0; i < point_size; ++i) {
                const auto x_ptr = reinterpret_cast<const double*>(&cloud.data[point_step * i + x_offset]);
                ret->points->data()[i] = PointType(x_ptr[0], x_ptr[1], x_ptr[2], 1.0f);
            }
        } else {
            for (size_t i = 0; i < point_size; ++i) {
                const auto x = reinterpret_cast<const double*>(&cloud.data[point_step * i + x_offset])[0];
                const auto y = reinterpret_cast<const double*>(&cloud.data[point_step * i + y_offset])[0];
                const auto z = reinterpret_cast<const double*>(&cloud.data[point_step * i + z_offset])[0];
                ret->points->data()[i] = PointType(x, y, z, 1.0f);
            }
        }
    }
    return ret;
}
sensor_msgs::msg::PointCloud2::UniquePtr toROS2msg(const sycl_points::PointCloudShared& cloud,
                                                   const std_msgs::msg::Header& header) {
    sensor_msgs::msg::PointCloud2::UniquePtr msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    msg->header = header;

    // add fields
    {
        sensor_msgs::msg::PointField field;
        field.count = 1;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.name = "x";
        field.offset = 0;
        msg->fields.push_back(field);
        field.name = "y";
        field.offset += sizeof(float);
        msg->fields.push_back(field);
        field.name = "z";
        field.offset += sizeof(float);
        msg->fields.push_back(field);
    }

    msg->width = cloud.size();
    msg->height = 1;
    msg->point_step = sizeof(float) * 3;
    msg->row_step = msg->point_step * cloud.size();
    msg->is_bigendian = false;
    msg->is_dense = true;

    msg->data.reserve(msg->row_step);
    for (size_t i = 0; i < cloud.size(); ++i) {
        auto& pt = cloud.points->data()[i];
        const auto pt_u8 = reinterpret_cast<uint8_t*>(pt.data());
        msg->data.insert(msg->data.end(), pt_u8, pt_u8 + sizeof(float) * 3);
    }
    return msg;
}
}  // namespace ros2
}  // namespace sycl_points