#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sycl_points/points/point_cloud.hpp>

namespace sycl_points {
namespace ros2 {

inline bool fromROS2msg(const sycl_points::sycl_utils::DeviceQueue& queue, const sensor_msgs::msg::PointCloud2& msg,
                        PointCloudShared::Ptr& cloud) {
    uint8_t x_type = 0;
    uint8_t y_type = 0;
    uint8_t z_type = 0;
    uint8_t rgb_type = 0;

    int32_t x_offset = -1;
    int32_t y_offset = -1;
    int32_t z_offset = -1;
    int32_t rgb_offset = -1;
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

    uint8_t* msg_data = sycl::malloc_device<uint8_t>(msg.data.size(), *queue.ptr);
    auto copy_event = queue.ptr->memcpy(msg_data, msg.data.data(), sizeof(uint8_t) * msg.data.size());

    if (x_type == sensor_msgs::msg::PointField::FLOAT32) {
        cloud->covs->clear();
        cloud->resize_points(point_size);
        queue.ptr
            ->submit([&](sycl::handler& h) {
                const size_t work_group_size = queue.get_work_group_size();
                const size_t global_size = queue.get_global_size(point_size);

                const uint8_t* msg_data_ptr = msg_data;
                PointType* ret_data_ptr = cloud->points->data();
                h.depends_on(copy_event);
                h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_id(0);
                    if (i >= point_size) return;
                    const auto x = reinterpret_cast<const float*>(&msg_data_ptr[point_step * i + x_offset])[0];
                    const auto y = reinterpret_cast<const float*>(&msg_data_ptr[point_step * i + y_offset])[0];
                    const auto z = reinterpret_cast<const float*>(&msg_data_ptr[point_step * i + z_offset])[0];
                    ret_data_ptr[i] = {x, y, z, 1.0f};
                });
            })
            .wait();
    } else if (x_type == sensor_msgs::msg::PointField::FLOAT64) {
        if (queue.is_supported_double()) {
            cloud->covs->clear();
            cloud->resize_points(point_size);
            queue.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = queue.get_work_group_size();
                    const size_t global_size = queue.get_global_size(point_size);

                    const uint8_t* msg_data_ptr = msg_data;
                    PointType* ret_data_ptr = cloud->points->data();
                    h.depends_on(copy_event);
                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= point_size) return;
                        const auto x = static_cast<float>(
                            reinterpret_cast<const double*>(&msg_data_ptr[point_step * i + x_offset])[0]);
                        const auto y = static_cast<float>(
                            reinterpret_cast<const double*>(&msg_data_ptr[point_step * i + y_offset])[0]);
                        const auto z = static_cast<float>(
                            reinterpret_cast<const double*>(&msg_data_ptr[point_step * i + z_offset])[0]);
                        ret_data_ptr[i] = {x, y, z, 1.0f};
                    });
                })
                .wait();
        } else {
            cloud->clear();
            cloud->points->reserve(point_size);
            for (size_t i = 0; i < point_size; ++i) {
                const auto x =
                    static_cast<float>(reinterpret_cast<const double*>(&msg.data[point_step * i + x_offset])[0]);
                const auto y =
                    static_cast<float>(reinterpret_cast<const double*>(&msg.data[point_step * i + y_offset])[0]);
                const auto z =
                    static_cast<float>(reinterpret_cast<const double*>(&msg.data[point_step * i + z_offset])[0]);
                cloud->points->emplace_back(x, y, z, 1.0f);
            }
        }
    }

    // RGB field
    if (rgb_offset >= 0) {
        if (rgb_type == sensor_msgs::msg::PointField::FLOAT32 || rgb_type == sensor_msgs::msg::PointField::UINT32) {
            cloud->resize_rgb(point_size);
            queue.ptr
                ->submit([&](sycl::handler& h) {
                    const size_t work_group_size = queue.get_work_group_size();
                    const size_t global_size = queue.get_global_size(point_size);

                    const uint8_t* msg_data_ptr = msg_data;
                    RGBType* ret_data_ptr = cloud->rgb->data();
                    h.depends_on(copy_event);
                    h.parallel_for(sycl::nd_range<1>(global_size, work_group_size), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_id(0);
                        if (i >= point_size) return;
                        const auto rgb = reinterpret_cast<const uint8_t*>(&msg_data_ptr[point_step * i + rgb_offset]);
                        ret_data_ptr[i].x() = rgb[0] / 255.0f;
                        ret_data_ptr[i].y() = rgb[1] / 255.0f;
                        ret_data_ptr[i].z() = rgb[2] / 255.0f;
                        ret_data_ptr[i].w() = rgb[3] / 255.0f;
                    });
                })
                .wait();
        } else {
            std::cerr << "Not supported rgb field type" << std::endl;
        }
    }
    sycl::free(msg_data, *queue.ptr);
    return true;
}
inline PointCloudShared::Ptr fromROS2msg(const sycl_points::sycl_utils::DeviceQueue& queue,
                                         const sensor_msgs::msg::PointCloud2& msg) {
    PointCloudShared::Ptr ret = nullptr;
    fromROS2msg(queue, msg, ret);
    return ret;
}

sensor_msgs::msg::PointCloud2::SharedPtr toROS2msg(const sycl_points::PointCloudShared& cloud,
                                                   const std_msgs::msg::Header& header) {
    sensor_msgs::msg::PointCloud2::SharedPtr msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    msg->header = header;

    const bool has_rgb = cloud.has_rgb();
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

        if (has_rgb) {
            field.name = "rgb";
            field.datatype = sensor_msgs::msg::PointField::UINT32;
            field.offset = 16;
            msg->fields.push_back(field);
        }
    }

    msg->width = cloud.size();
    msg->height = 1;
    msg->point_step = sizeof(PointType);
    if (has_rgb) {
        msg->point_step += sizeof(float);
    }
    msg->row_step = msg->point_step * cloud.size();
    msg->is_bigendian = false;
    msg->is_dense = true;

    msg->data.reserve(msg->row_step);

    if (has_rgb) {
        // point + rgb
        for (size_t i = 0; i < cloud.size(); ++i) {
            const auto& pt = cloud.points->data()[i];
            const auto pt_u8 = reinterpret_cast<const uint8_t*>(pt.data());
            msg->data.insert(msg->data.end(), pt_u8, pt_u8 + sizeof(PointType));

            const Eigen::Vector<uint8_t, 4> rgb =
                (cloud.rgb->data()[i].array().min(1.0f).max(0.0f) * 255.0f).cast<uint8_t>();
            const auto rgb_ptr = reinterpret_cast<const uint8_t*>(rgb.data());
            msg->data.insert(msg->data.end(), rgb_ptr, rgb_ptr + sizeof(uint8_t) * 4);
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
