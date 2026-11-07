import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_name = "sycl_points_ros2"
    node_name = "prior_map_localization"
    config_path = os.path.join(
        get_package_share_directory(package_name), "config", "prior_map_localization.yaml"
    )
    with open(config_path, encoding="utf-8") as config_file:
        params = yaml.safe_load(config_file)[node_name]["ros__parameters"]

    launch_args = [
        DeclareLaunchArgument(
            "map/path", description="Path to the prior map PCD or PLY file"
        ),
        DeclareLaunchArgument(
            "odom_topic",
            default_value="sycl_lio/odom",
            description="Input odometry topic",
        ),
        DeclareLaunchArgument(
            "submap_topic",
            default_value="sycl_lio/submap",
            description="Input submap point cloud topic",
        ),
    ]
    node_params = {"map/path": LaunchConfiguration("map/path")}
    for name, value in params.items():
        launch_args.append(
            DeclareLaunchArgument(name, default_value=str(value), description="")
        )
        node_params[name] = ParameterValue(
            LaunchConfiguration(name), value_type=type(value)
        )

    localization = Node(
        package=package_name,
        executable="prior_map_localization_node",
        name=node_name,
        output="screen",
        parameters=[node_params],
        remappings=[
            ("sycl_lio/odom", LaunchConfiguration("odom_topic")),
            ("sycl_lio/submap", LaunchConfiguration("submap_topic")),
        ],
    )
    return LaunchDescription(launch_args + [localization])
