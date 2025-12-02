#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace ros2 {

inline bool fromROS2msg(const sycl_points::sycl_utils::DeviceQueue& queue, const sensor_msgs::msg::PointCloud2& msg,
                        PointCloudShared::Ptr& cloud, sycl_points::shared_vector_ptr<uint8_t>& msg_buffer) {
    uint8_t x_type = 0;
    uint8_t y_type = 0;
    uint8_t z_type = 0;
    uint8_t rgb_type = 0;
    uint8_t intensity_type = 0;

    int32_t x_offset = -1;
    int32_t y_offset = -1;
    int32_t z_offset = -1;
    int32_t rgb_offset = -1;
    int32_t intensity_offset = -1;

    for (const auto& field : msg.fields) {
        if (field.name == "x") {
            x_type = field.datatype;
            x_offset = field.offset;
        } else if (field.name == "y") {
            y_type = field.datatype;
            y_offset = field.offset;
        } else if (field.name == "z") {
            z_type = field.datatype;
            z_offset = field.offset;
        } else if (field.name == "rgb" || field.name == "rgba") {
            rgb_type = field.datatype;
            rgb_offset = field.offset;
        } else if (field.name == "intensity") {
            intensity_type = field.datatype;
            intensity_offset = field.offset;
        }
    }
    if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
        std::cerr << "Not found point cloud field" << std::endl;
        return false;
    }
    if ((x_type != sensor_msgs::msg::PointField::FLOAT32 && x_type != sensor_msgs::msg::PointField::FLOAT64)) {
        std::cerr << "Not supported point field type" << std::endl;
        return false;
    }
    if (x_type != y_type || x_type != z_type) {
        std::cerr << "Not supported point field type" << std::endl;
        return false;
    }

    if (cloud == nullptr || *cloud->queue.ptr != *queue.ptr) {
        cloud = std::make_shared<PointCloudShared>(queue);
    }

    const size_t point_step = msg.point_step;
    const size_t point_size = msg.width * msg.height;

    if (point_size == 0) {
        cloud->clear();
        return true;
    }

    const bool has_rgb_field = (rgb_offset >= 0) && (rgb_type == sensor_msgs::msg::PointField::FLOAT32 ||
                                                     rgb_type == sensor_msgs::msg::PointField::UINT32);
    if (rgb_offset >= 0 && !has_rgb_field) {
        std::cerr << "Not supported rgb field type" << std::endl;
    }

    const bool has_intensity_field = (intensity_offset >= 0) && intensity_type == sensor_msgs::msg::PointField::FLOAT32;
    if (intensity_offset >= 0 && !has_intensity_field) {
        std::cerr << "Not supported intensity field type" << std::endl;
    }

    auto submit_device_conversion = [&](auto scalar_tag) {
        using ScalarT = decltype(scalar_tag);

        if (msg_buffer->size() < msg.data.size()) {
            msg_buffer->resize(msg.data.size());
        }
        uint8_t* msg_data = msg_buffer->data();
        auto copy_event = queue.ptr->memcpy(msg_data, msg.data.data(), sizeof(uint8_t) * msg.data.size());

        sycl_utils::events events;
        events += queue.ptr->submit([&](sycl::handler& h) {
            const size_t work_group_size = queue.get_work_group_size();
            const size_t global_size = queue.get_global_size(point_size);

            const uint8_t* msg_data_ptr = msg_data;
            PointType* points_ptr = cloud->points->data();
            RGBType* rgb_ptr = has_rgb_field ? cloud->rgb->data() : nullptr;
            float* intensity_ptr = has_intensity_field ? cloud->intensities->data() : nullptr;
            const bool write_rgb = has_rgb_field;
            const bool write_intensity = has_intensity_field;

            h.depends_on(copy_event);
            h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                const size_t i = item.get_global_id(0);
                if (i >= point_size) return;
                const size_t base = point_step * i;
                const float x = static_cast<float>(reinterpret_cast<const ScalarT*>(&msg_data_ptr[base + x_offset])[0]);
                const float y = static_cast<float>(reinterpret_cast<const ScalarT*>(&msg_data_ptr[base + y_offset])[0]);
                const float z = static_cast<float>(reinterpret_cast<const ScalarT*>(&msg_data_ptr[base + z_offset])[0]);
                points_ptr[i] = {x, y, z, 1.0f};

                if (write_rgb) {
                    const auto rgb = reinterpret_cast<const uint8_t*>(&msg_data_ptr[base + rgb_offset]);
                    auto& dst = rgb_ptr[i];
                    dst.x() = static_cast<float>(rgb[0]) / 255.0f;
                    dst.y() = static_cast<float>(rgb[1]) / 255.0f;
                    dst.z() = static_cast<float>(rgb[2]) / 255.0f;
                    dst.w() = static_cast<float>(rgb[3]) / 255.0f;
                }

                if (write_intensity) {
                    intensity_ptr[i] = reinterpret_cast<const float*>(&msg_data_ptr[base + intensity_offset])[0];
                }
            });
        });
        events.wait_and_throw();
    };

    // prepare container
    {
        cloud->clear();
        cloud->resize_points(point_size);
        if (has_rgb_field) {
            cloud->resize_rgb(point_size);
        }
        if (has_intensity_field) {
            cloud->resize_intensities(point_size);
        }
    };

    if (x_type == sensor_msgs::msg::PointField::FLOAT32) {
        submit_device_conversion(float{});
        return true;
    }

    if (x_type == sensor_msgs::msg::PointField::FLOAT64) {
        if (queue.is_supported_double()) {
            submit_device_conversion(double{});
            return true;
        }

        for (size_t i = 0; i < point_size; ++i) {
            const size_t base = point_step * i;

            {
                const auto x = static_cast<float>(reinterpret_cast<const double*>(&msg.data[base + x_offset])[0]);
                const auto y = static_cast<float>(reinterpret_cast<const double*>(&msg.data[base + y_offset])[0]);
                const auto z = static_cast<float>(reinterpret_cast<const double*>(&msg.data[base + z_offset])[0]);
                auto& dst = (*cloud->points)[i];
                dst.x() = x;
                dst.y() = y;
                dst.z() = z;
                dst.w() = 1.0f;
            }

            if (has_rgb_field) {
                const auto rgb = reinterpret_cast<const uint8_t*>(&msg.data[base + rgb_offset]);
                auto& dst = (*cloud->rgb)[i];
                dst.x() = static_cast<float>(rgb[0]) / 255.0f;
                dst.y() = static_cast<float>(rgb[1]) / 255.0f;
                dst.z() = static_cast<float>(rgb[2]) / 255.0f;
                dst.w() = static_cast<float>(rgb[3]) / 255.0f;
            }

            if (has_intensity_field) {
                const auto intensity = reinterpret_cast<const float*>(&msg.data[base + intensity_offset])[0];
                (*cloud->intensities)[i] = intensity;
            }
        }
        return true;
    }

    return false;
}

inline PointCloudShared::Ptr fromROS2msg(const sycl_points::sycl_utils::DeviceQueue& queue,
                                         const sensor_msgs::msg::PointCloud2& msg,
                                         sycl_points::shared_vector_ptr<uint8_t>& msg_buffer) {
    PointCloudShared::Ptr ret = nullptr;
    fromROS2msg(queue, msg, ret, msg_buffer);
    return ret;
}

sensor_msgs::msg::PointCloud2::SharedPtr toROS2msg(const sycl_points::PointCloudShared& cloud,
                                                   const std_msgs::msg::Header& header) {
    sensor_msgs::msg::PointCloud2::SharedPtr msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    msg->header = header;

    const bool has_rgb = cloud.has_rgb();
    const bool has_intensity = cloud.has_intensity();

    // add fields
    {
        sensor_msgs::msg::PointField field;
        field.count = 1;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field.name = "x";
        field.offset = 0 * sizeof(float);  // 0;
        msg->fields.push_back(field);
        field.name = "y";
        field.offset = 1 * sizeof(float);  // 4
        msg->fields.push_back(field);
        field.name = "z";
        field.offset = 2 * sizeof(float);  // 8
        msg->fields.push_back(field);
        field.offset = 3 * sizeof(float);  // 12

        if (has_rgb) {
            field.name = "rgb";
            field.datatype = sensor_msgs::msg::PointField::UINT32;
            field.offset += sizeof(float);
            msg->fields.push_back(field);
        }
        if (has_intensity) {
            field.name = "intensity";
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.offset += sizeof(float);
            msg->fields.push_back(field);
        }
    }

    msg->width = cloud.size();
    msg->height = 1;
    msg->point_step = sizeof(PointType);
    if (has_rgb) {
        msg->point_step += sizeof(float);
    }
    if (has_intensity) {
        msg->point_step += sizeof(float);
    }
    msg->row_step = msg->point_step * cloud.size();
    msg->is_bigendian = false;
    msg->is_dense = true;

    msg->data.reserve(msg->row_step);

    if (has_rgb || has_intensity) {
        // point + (rgb and/or intensity)
        for (size_t i = 0; i < cloud.size(); ++i) {
            const auto& pt = cloud.points->data()[i];
            const auto pt_u8 = reinterpret_cast<const uint8_t*>(pt.data());
            msg->data.insert(msg->data.end(), pt_u8, pt_u8 + sizeof(PointType));

            if (has_rgb) {
                const Eigen::Vector<uint8_t, 4> rgb =
                    (cloud.rgb->data()[i].array().min(1.0f).max(0.0f) * 255.0f).cast<uint8_t>();
                const auto rgb_ptr = reinterpret_cast<const uint8_t*>(rgb.data());
                msg->data.insert(msg->data.end(), rgb_ptr, rgb_ptr + sizeof(uint8_t) * 4);
            }
            if (has_intensity) {
                const float intensity = cloud.intensities->data()[i];
                const auto intensity_ptr = reinterpret_cast<const uint8_t*>(&intensity);
                msg->data.insert(msg->data.end(), intensity_ptr, intensity_ptr + sizeof(float));
            }
        }
    } else {
        // point only
        for (size_t i = 0; i < cloud.size(); ++i) {
            const auto& pt = cloud.points->data()[i];
            const auto pt_u8 = reinterpret_cast<const uint8_t*>(pt.data());
            msg->data.insert(msg->data.end(), pt_u8, pt_u8 + sizeof(PointType));
        }
    }

    return msg;
}
}  // namespace ros2
}  // namespace sycl_points
