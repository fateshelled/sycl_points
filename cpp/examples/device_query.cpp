#include "sycl_points/utils/sycl_utils.hpp"

int main(int argc, char** argv) {
    for (auto platform : sycl::platform::get_platforms()) {
        for (auto device : platform.get_devices()) {
            sycl_points::sycl_utils::print_device_info(device);
        }
    }

    return 0;
}
