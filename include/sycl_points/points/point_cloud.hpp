#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <sycl_points/points/types.hpp>

namespace sycl_points {

/// @brief CPU point cloud class with point and covariance container.
struct PointCloudCPU {
    using Ptr = std::shared_ptr<PointCloudCPU>;
    using ConstPtr = std::shared_ptr<PointCloudCPU>;

    /// @brief point container
    std::shared_ptr<PointContainerCPU> points = nullptr;
    /// @brief covariance container
    std::shared_ptr<CovarianceContainerCPU> covs = nullptr;
    /// @brief normal container
    std::shared_ptr<NormalContainerCPU> normals = nullptr;
    /// @brief rgb container
    std::shared_ptr<RGBContainerCPU> rgb = nullptr;
    /// @brief color gradient container
    std::shared_ptr<ColorGradientContainerCPU> color_gradients = nullptr;
    /// @brief intensity gradient container
    std::shared_ptr<IntensityGradientContainerCPU> intensity_gradients = nullptr;
    /// @brief intensity container
    std::shared_ptr<IntensityContainerCPU> intensities = nullptr;
    /// @brief timestamp offset container
    std::shared_ptr<TimestampContainerCPU> timestamp_offsets = nullptr;
    /// @brief timestamp start time in milliseconds
    double start_time_ms = 0.0;
    /// @brief timestamp end time in milliseconds
    double end_time_ms = 0.0;

    /// @brief Constructor
    PointCloudCPU() {
        this->points = std::make_shared<PointContainerCPU>();
        this->covs = std::make_shared<CovarianceContainerCPU>();
        this->normals = std::make_shared<NormalContainerCPU>();
        this->rgb = std::make_shared<RGBContainerCPU>();
        this->color_gradients = std::make_shared<ColorGradientContainerCPU>();
        this->intensity_gradients = std::make_shared<IntensityGradientContainerCPU>();
        this->intensities = std::make_shared<IntensityContainerCPU>();
        this->timestamp_offsets = std::make_shared<TimestampContainerCPU>();
    }

    /// @brief number of point
    size_t size() const { return this->points->size(); }
    /// @brief has covariance field or not
    bool has_cov() const { return this->covs != nullptr && this->covs->size() == this->points->size(); }
    /// @brief has normal field or not
    bool has_normal() const { return this->normals != nullptr && this->normals->size() == this->points->size(); }
    /// @brief has RGB field or not
    bool has_rgb() const { return this->rgb != nullptr && this->rgb->size() == this->points->size(); }
    /// @brief has color gradient field or not
    bool has_color_gradient() const {
        return this->color_gradients != nullptr && this->color_gradients->size() == this->points->size();
    }
    /// @brief has intensity gradient field or not
    bool has_intensity_gradient() const {
        return this->intensity_gradients != nullptr && this->intensity_gradients->size() == this->points->size();
    }
    /// @brief has intensity field or not
    bool has_intensity() const {
        return this->intensities != nullptr && this->intensities->size() == this->points->size();
    }
    /// @brief has timestamp field or not
    bool has_timestamps() const {
        return this->timestamp_offsets != nullptr && this->timestamp_offsets->size() == this->points->size() &&
               !this->timestamp_offsets->empty();
    }

    /// @brief Update the end timestamp based on available offsets.
    void update_end_time() {
        if (this->timestamp_offsets && !this->timestamp_offsets->empty()) {
            const auto max_offset = *std::max_element(this->timestamp_offsets->begin(), this->timestamp_offsets->end());
            this->end_time_ms = this->start_time_ms + static_cast<double>(max_offset);
        } else {
            this->end_time_ms = this->start_time_ms;
        }
    }
};

/// @brief Shared memory point cloud class with point and covariance container.
struct PointCloudShared {
    using Ptr = std::shared_ptr<PointCloudShared>;
    using ConstPtr = std::shared_ptr<PointCloudShared>;

    /// @brief SYCL queue
    sycl_utils::DeviceQueue queue;
    /// @brief point container
    std::shared_ptr<PointContainerShared> points = nullptr;
    /// @brief covariance container
    std::shared_ptr<CovarianceContainerShared> covs = nullptr;
    /// @brief normal container
    std::shared_ptr<NormalContainerShared> normals = nullptr;
    /// @brief rgb container
    std::shared_ptr<RGBContainerShared> rgb = nullptr;
    /// @brief color gradient container
    std::shared_ptr<ColorGradientContainerShared> color_gradients = nullptr;
    /// @brief intensity gradient container
    std::shared_ptr<IntensityGradientContainerShared> intensity_gradients = nullptr;
    /// @brief intensity container
    std::shared_ptr<IntensityContainerShared> intensities = nullptr;
    /// @brief timestamp offset container
    std::shared_ptr<TimestampContainerShared> timestamp_offsets = nullptr;
    /// @brief timestamp start time in milliseconds
    double start_time_ms = 0.0;
    /// @brief timestamp end time in milliseconds
    double end_time_ms = 0.0;

    /// @brief Constructor
    /// @param queue SYCL queue
    PointCloudShared(const sycl_utils::DeviceQueue& q) : queue(q) {
        this->points = std::make_shared<PointContainerShared>(0, *this->queue.ptr);
        this->covs = std::make_shared<CovarianceContainerShared>(0, *this->queue.ptr);
        this->normals = std::make_shared<NormalContainerShared>(0, *this->queue.ptr);
        this->rgb = std::make_shared<RGBContainerShared>(0, *this->queue.ptr);
        this->color_gradients = std::make_shared<ColorGradientContainerShared>(0, *this->queue.ptr);
        this->intensity_gradients = std::make_shared<IntensityGradientContainerShared>(0, *this->queue.ptr);
        this->intensities = std::make_shared<IntensityContainerShared>(0, *this->queue.ptr);
        this->timestamp_offsets = std::make_shared<TimestampContainerShared>(0, *this->queue.ptr);
    }

    /// @brief Copy from CPU point cloud
    /// @param queue SYCL queue
    /// @param cpu CPU point cloud
    PointCloudShared(const sycl_utils::DeviceQueue& q, const PointCloudCPU& cpu) : queue(q) {
        const size_t N = cpu.size();

        sycl_utils::events copy_events;
        const bool is_cpu = sycl_utils::is_cpu(*this->queue.ptr);

        if (cpu.has_cov()) {
            this->covs = std::make_shared<CovarianceContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->covs->data(), cpu.covs->data(), N * sizeof(Covariance));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->covs->data()[i] = cpu.covs->data()[i];
                }
            }
        } else {
            this->covs = std::make_shared<CovarianceContainerShared>(0, *this->queue.ptr);
        }

        if (cpu.has_normal()) {
            this->normals = std::make_shared<NormalContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->normals->data(), cpu.normals->data(), N * sizeof(Normal));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->normals->data()[i] = cpu.normals->data()[i];
                }
            }
        } else {
            this->normals = std::make_shared<NormalContainerShared>(0, *this->queue.ptr);
        }

        if (cpu.has_rgb()) {
            this->rgb = std::make_shared<RGBContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->rgb->data(), cpu.rgb->data(), N * sizeof(RGBType));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->rgb->data()[i] = cpu.rgb->data()[i];
                }
            }
        } else {
            this->rgb = std::make_shared<RGBContainerShared>(0, *this->queue.ptr);
        }

        if (cpu.has_color_gradient()) {
            this->color_gradients = std::make_shared<ColorGradientContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->color_gradients->data(), cpu.color_gradients->data(),
                                                       N * sizeof(ColorGradient));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->color_gradients->data()[i] = cpu.color_gradients->data()[i];
                }
            }
        } else {
            this->color_gradients = std::make_shared<ColorGradientContainerShared>(0, *this->queue.ptr);
        }

        if (cpu.has_intensity_gradient()) {
            this->intensity_gradients = std::make_shared<IntensityGradientContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events +=
                    this->queue.ptr->memcpy(this->intensity_gradients->data(), cpu.intensity_gradients->data(),
                                             N * sizeof(IntensityGradient));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->intensity_gradients->data()[i] = cpu.intensity_gradients->data()[i];
                }
            }
        } else {
            this->intensity_gradients = std::make_shared<IntensityGradientContainerShared>(0, *this->queue.ptr);
        }

        if (cpu.has_intensity()) {
            this->intensities = std::make_shared<IntensityContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events +=
                    this->queue.ptr->memcpy(this->intensities->data(), cpu.intensities->data(), N * sizeof(float));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->intensities->data()[i] = cpu.intensities->data()[i];
                }
            }
        } else {
            this->intensities = std::make_shared<IntensityContainerShared>(0, *this->queue.ptr);
        }

        if (cpu.has_timestamps()) {
            this->timestamp_offsets = std::make_shared<TimestampContainerShared>(N, *this->queue.ptr);
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->timestamp_offsets->data(), cpu.timestamp_offsets->data(),
                                                       N * sizeof(TimestampOffset));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->timestamp_offsets->data()[i] = cpu.timestamp_offsets->data()[i];
                }
            }
            this->start_time_ms = cpu.start_time_ms;
            this->end_time_ms = cpu.end_time_ms;
        } else {
            this->timestamp_offsets = std::make_shared<TimestampContainerShared>(0, *this->queue.ptr);
            this->start_time_ms = 0.0;
            this->end_time_ms = 0.0;
        }

        this->points = std::make_shared<PointContainerShared>(N, *this->queue.ptr);
        if (N > 0) {
            if (is_cpu) {
                copy_events += this->queue.ptr->memcpy(this->points->data(), cpu.points->data(), N * sizeof(PointType));
            } else {
                for (size_t i = 0; i < N; ++i) {
                    this->points->data()[i] = cpu.points->data()[i];
                }
            }
        }
        copy_events.wait_and_throw();
    }

    /// @brief Copy from other shared point cloud using the same queue as the source.
    /// @param other shared point cloud
    PointCloudShared(const PointCloudShared& other) : PointCloudShared(other.queue, other) {}

    /// @brief Copy from other shared point cloud onto the specified queue
    /// @param target_queue destination queue
    /// @param other shared point cloud
    PointCloudShared(const sycl_utils::DeviceQueue& target_queue, const PointCloudShared& other) : queue(target_queue) {
        if (!this->queue.ptr) {
            throw std::runtime_error("Target queue is not initialised");
        }
        if (!other.queue.ptr) {
            throw std::runtime_error("Source point cloud queue is not initialised");
        }
        if (!other.points) {
            throw std::runtime_error("Source point cloud points are not initialised");
        }

        const size_t N = other.size();
        const bool same_context = this->queue.ptr->get_context() == other.queue.ptr->get_context();
        sycl_utils::events copy_events;
        sycl::queue& copy_queue = *other.queue.ptr;

        this->copy_attribute(this->covs, other.covs, N, other.has_cov(), same_context, copy_queue, copy_events);
        this->copy_attribute(this->normals, other.normals, N, other.has_normal(), same_context, copy_queue, copy_events);
        this->copy_attribute(this->rgb, other.rgb, N, other.has_rgb(), same_context, copy_queue, copy_events);
        this->copy_attribute(this->color_gradients, other.color_gradients, N, other.has_color_gradient(), same_context,
                             copy_queue, copy_events);
        this->copy_attribute(this->intensity_gradients, other.intensity_gradients, N, other.has_intensity_gradient(),
                             same_context, copy_queue, copy_events);
        this->copy_attribute(this->intensities, other.intensities, N, other.has_intensity(), same_context, copy_queue,
                             copy_events);
        this->copy_attribute(this->points, other.points, N, true, same_context, copy_queue, copy_events);
        this->copy_attribute(this->timestamp_offsets, other.timestamp_offsets, N, other.has_timestamps(), same_context,
                             copy_queue, copy_events);
        this->start_time_ms = other.has_timestamps() ? other.start_time_ms : 0.0;
        this->end_time_ms = other.has_timestamps() ? other.end_time_ms : 0.0;

        copy_events.wait_and_throw();
    }

    /// @brief destructor
    ~PointCloudShared() {}

    /// @brief number of points
    size_t size() const { return this->points->size(); }
    /// @brief has covariance field or not
    bool has_cov() const { return this->covs->size() > 0 && this->covs->size() == this->points->size(); }
    /// @brief has normal field or not
    bool has_normal() const { return this->normals->size() > 0 && this->normals->size() == this->points->size(); }
    /// @brief has RGB field or not
    bool has_rgb() const { return this->rgb->size() > 0 && this->rgb->size() == this->points->size(); }
    bool has_color_gradient() const {
        return this->color_gradients->size() > 0 && this->color_gradients->size() == this->points->size();
    }
    bool has_intensity_gradient() const {
        return this->intensity_gradients->size() > 0 && this->intensity_gradients->size() == this->points->size();
    }
    /// @brief has intensity field or not
    bool has_intensity() const {
        return this->intensities->size() > 0 && this->intensities->size() == this->points->size();
    }
    /// @brief has timestamp field or not
    bool has_timestamps() const {
        return this->timestamp_offsets->size() > 0 && this->timestamp_offsets->size() == this->points->size();
    }

    /// @brief pointer of points
    PointType* points_ptr() const { return this->points->data(); }
    /// @brief pointer of covariances
    Covariance* covs_ptr() const { return this->covs->data(); }
    /// @brief pointer of normals
    Normal* normals_ptr() const { return this->normals->data(); }
    /// @brief pointer of RGB
    RGBType* rgb_ptr() const { return this->rgb->data(); }
    /// @brief pointer of color gradients
    ColorGradient* color_gradients_ptr() const { return this->color_gradients->data(); }
    /// @brief pointer of intensity gradients
    IntensityGradient* intensity_gradients_ptr() const { return this->intensity_gradients->data(); }
    /// @brief pointer of intensity
    float* intensities_ptr() const { return this->intensities->data(); }
    /// @brief pointer of timestamp offsets
    TimestampOffset* timestamp_offsets_ptr() const { return this->timestamp_offsets->data(); }

    /// @brief resize point container
    /// @param N size
    void resize_points(size_t N) const { this->points->resize(N); }
    /// @brief resize covariance container
    /// @param N size
    void resize_covs(size_t N) const { this->covs->resize(N); }
    /// @brief resize normal container
    /// @param N size
    void resize_normals(size_t N) const { this->normals->resize(N); }
    /// @brief resize RGB container
    /// @param N size
    void resize_rgb(size_t N) const { this->rgb->resize(N); }
    /// @brief resize color gradient container
    /// @param N size
    void resize_color_gradients(size_t N) const { this->color_gradients->resize(N); }
    /// @brief resize intensity gradient container
    /// @param N size
    void resize_intensity_gradients(size_t N) const { this->intensity_gradients->resize(N); }
    /// @brief resize intensity container
    /// @param N size
    void resize_intensities(size_t N) const { this->intensities->resize(N); }
    /// @brief resize timestamp container
    void resize_timestamps(size_t N) const { this->timestamp_offsets->resize(N); }

    /// @brief reserve point container
    /// @param N size
    void reserve_points(size_t N) const { this->points->reserve(N); }
    /// @brief reserve covariance container
    /// @param N size
    void reserve_covs(size_t N) const { this->covs->reserve(N); }
    /// @brief reserve normal container
    /// @param N size
    void reserve_normals(size_t N) const { this->normals->reserve(N); }
    /// @brief reserve RGB container
    /// @param N size
    void reserve_rgb(size_t N) const { this->rgb->reserve(N); }
    /// @brief reserve color gradient container
    /// @param N size
    void reserve_color_gradients(size_t N) const { this->color_gradients->reserve(N); }
    /// @brief reserve intensity gradient container
    /// @param N size
    void reserve_intensity_gradients(size_t N) const { this->intensity_gradients->reserve(N); }
    /// @brief reserve intensity container
    /// @param N size
    void reserve_intensities(size_t N) const { this->intensities->reserve(N); }
    /// @brief reserve timestamp container
    void reserve_timestamps(size_t N) const { this->timestamp_offsets->reserve(N); }

    /// @brief Erase all points and covariances data.
    void clear() {
        this->points->clear();
        this->covs->clear();
        this->normals->clear();
        this->rgb->clear();
        this->color_gradients->clear();
        this->intensity_gradients->clear();
        this->intensities->clear();
        this->timestamp_offsets->clear();
        this->start_time_ms = 0.0;
        this->end_time_ms = 0.0;
    }

    void extend(const PointCloudShared& other) {
        const size_t org_size = this->size();

        this->points->reserve(org_size + other.size());
        if (this->has_cov() && other.has_cov()) {
            this->covs->insert(this->covs->end(), other.covs->begin(), other.covs->end());
        }
        if (this->has_normal() && other.has_normal()) {
            this->normals->insert(this->normals->end(), other.normals->begin(), other.normals->end());
        }
        if (this->has_rgb() && other.has_rgb()) {
            this->rgb->insert(this->rgb->end(), other.rgb->begin(), other.rgb->end());
        }
        if (this->has_color_gradient() && other.has_color_gradient()) {
            this->color_gradients->insert(this->color_gradients->end(), other.color_gradients->begin(),
                                          other.color_gradients->end());
        }
        if (this->has_intensity_gradient() && other.has_intensity_gradient()) {
            this->intensity_gradients->insert(this->intensity_gradients->end(), other.intensity_gradients->begin(),
                                              other.intensity_gradients->end());
        }
        if (this->has_intensity() && other.has_intensity()) {
            this->intensities->insert(this->intensities->end(), other.intensities->begin(), other.intensities->end());
        }
        this->points->insert(this->points->end(), other.points->begin(), other.points->end());

        this->merge_timestamp_offsets(other, org_size);
    }

    void erase(size_t start_idx, size_t end_idx) {
        if (this->has_cov()) {
            this->covs->erase(this->covs->begin() + start_idx, this->covs->begin() + end_idx);
        }
        if (this->has_normal()) {
            this->normals->erase(this->normals->begin() + start_idx, this->normals->begin() + end_idx);
        }
        if (this->has_rgb()) {
            this->rgb->erase(this->rgb->begin() + start_idx, this->rgb->begin() + end_idx);
        }
        if (this->has_color_gradient()) {
            this->color_gradients->erase(this->color_gradients->begin() + start_idx,
                                         this->color_gradients->begin() + end_idx);
        }
        if (this->has_intensity_gradient()) {
            this->intensity_gradients->erase(this->intensity_gradients->begin() + start_idx,
                                             this->intensity_gradients->begin() + end_idx);
        }
        if (this->has_intensity()) {
            this->intensities->erase(this->intensities->begin() + start_idx, this->intensities->begin() + end_idx);
        }
        if (this->has_timestamps()) {
            this->timestamp_offsets->erase(this->timestamp_offsets->begin() + start_idx,
                                           this->timestamp_offsets->begin() + end_idx);
            if (this->timestamp_offsets->empty()) {
                this->start_time_ms = 0.0;
                this->end_time_ms = 0.0;
            } else {
                const auto max_offset =
                    *std::max_element(this->timestamp_offsets->begin(), this->timestamp_offsets->end());
                this->end_time_ms = this->start_time_ms + static_cast<double>(max_offset);
            }
        }
        this->points->erase(this->points->begin() + start_idx, this->points->begin() + end_idx);
    }

    void operator+=(const PointCloudShared& pc) { this->extend(pc); }

private:
    template <typename Container>
    void copy_attribute(std::shared_ptr<Container>& dest, const std::shared_ptr<Container>& src, size_t count,
                        bool should_copy, bool same_context, sycl::queue& copy_queue,
                        sycl_utils::events& copy_events) {
        const size_t allocation_size = should_copy ? count : 0;
        dest = std::make_shared<Container>(allocation_size, *this->queue.ptr);

        if (!should_copy || count == 0) {
            return;
        }

        if (!src) {
            return;
        }

        if (same_context) {
            copy_events += copy_queue.memcpy(dest->data(), src->data(), count * sizeof(typename Container::value_type));
        } else {
            dest->assign(src->begin(), src->end());
        }
    }

    /// @brief Reset timestamp information when inconsistent combinations are requested.
    void invalidate_timestamps() {
        this->timestamp_offsets->clear();
        this->start_time_ms = 0.0;
        this->end_time_ms = 0.0;
    }

    /// @brief Shift the local timestamp base towards an earlier value.
    void shift_timestamp_base(double new_start_time_ms) {
        if (!this->has_timestamps() || new_start_time_ms >= this->start_time_ms) {
            if (new_start_time_ms > this->start_time_ms) {
                // Later bases would require negative offsets which are unsupported.
                this->invalidate_timestamps();
            }
            return;
        }

        const double delta_ms = this->start_time_ms - new_start_time_ms;
        const double max_value = static_cast<double>(std::numeric_limits<TimestampOffset>::max());

        for (auto& offset : *this->timestamp_offsets) {
            const double adjusted_offset = static_cast<double>(offset) + delta_ms;
            if (adjusted_offset > max_value) {
                throw std::runtime_error("Timestamp offset overflow while shifting base");
            }
            offset = static_cast<TimestampOffset>(adjusted_offset);
        }

        this->start_time_ms = new_start_time_ms;
    }

    /// @brief Merge timestamp offsets when extending point clouds.
    void merge_timestamp_offsets(const PointCloudShared& other, size_t original_size) {
        if (other.size() == 0) {
            return;
        }

        if (!other.has_timestamps()) {
            if (this->has_timestamps()) {
                this->invalidate_timestamps();
            }
            return;
        }

        if (!this->has_timestamps()) {
            if (original_size == 0) {
                // This cloud was empty, so we can just adopt the timestamps from the other cloud.
                this->timestamp_offsets->insert(this->timestamp_offsets->end(), other.timestamp_offsets->begin(),
                                                other.timestamp_offsets->end());
                this->start_time_ms = other.start_time_ms;
                this->end_time_ms = other.end_time_ms;
            } else {
                // This cloud has points but no timestamps. The merged cloud will also not have timestamps
                // to maintain consistency. Timestamps from `other` are intentionally dropped.
                // No action is needed as `this->timestamp_offsets` is already empty.
            }
            return;
        }

        const double new_start_ms = std::min(this->start_time_ms, other.start_time_ms);
        if (new_start_ms < this->start_time_ms) {
            this->shift_timestamp_base(new_start_ms);
        }

        const double base_delta_ms = other.start_time_ms - new_start_ms;
        const double max_value = static_cast<double>(std::numeric_limits<TimestampOffset>::max());
        if (base_delta_ms > max_value) {
            throw std::runtime_error("Timestamp base delta exceeds representable offset range");
        }

        this->timestamp_offsets->reserve(this->timestamp_offsets->size() + other.timestamp_offsets->size());
        for (const auto offset : *other.timestamp_offsets) {
            const double adjusted_offset = static_cast<double>(offset) + base_delta_ms;
            if (adjusted_offset > max_value) {
                throw std::runtime_error("Timestamp offset overflow while merging clouds");
            }
            this->timestamp_offsets->push_back(static_cast<TimestampOffset>(adjusted_offset));
        }
        this->start_time_ms = new_start_ms;
        this->end_time_ms = std::max(this->end_time_ms, other.end_time_ms);
    }
};

}  // namespace sycl_points
