from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def declare_params_from_yaml(yaml_path: str, target_node="lidar_inertial_odometry_node"):
    launch_args = []
    node_args = {}
    with open(yaml_path, "r") as f:
        all_params = yaml.safe_load(f)

    for node_name in all_params.keys():
        if node_name == target_node:
            node_params: dict = all_params[node_name]["ros__parameters"]
            for name, value in node_params.items():
                if isinstance(value, float):
                    value_str = format(value, "f")
                else:
                    value_str = str(value)
                launch_args.append(
                    DeclareLaunchArgument(name, default_value=value_str, description="")
                )
                node_args[name] = LaunchConfiguration(name)
            break
    return launch_args, node_args


def generate_launch_description():
    package_name = "sycl_points_ros2"
    package_dir = get_package_share_directory(package_name)
    param_yaml = os.path.join(package_dir, "config", "lidar_inertial_odometry.yaml")
    launch_args, node_args = declare_params_from_yaml(param_yaml, "lidar_inertial_odometry_node")

    launch_args.extend(
        [
            DeclareLaunchArgument(
                "rosbag/uri", default_value="", description="input rosbag path"
            ),
            DeclareLaunchArgument(
                "rosbag/start_offset/sec",
                default_value="0.0",
                description="bag start offset in seconds",
            ),
            DeclareLaunchArgument(
                "eval/output_tum",
                default_value="sycl_lio_odom.tum",
                description="output tum filepath",
            ),
            DeclareLaunchArgument(
                "eval/write_first_frame",
                default_value="true",
                choices=["true", "false"],
                description="write first frame pose to tum",
            ),
            DeclareLaunchArgument(
                "eval/exit_on_end",
                default_value="true",
                choices=["true", "false"],
                description="shutdown node after evaluation",
            ),
        ]
    )

    nodes = [
        Node(
            package=package_name,
            executable="lidar_inertial_odometry_bag_eval_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                node_args,
                {
                    "rosbag/uri": LaunchConfiguration("rosbag/uri"),
                    "rosbag/start_offset/sec": ParameterValue(
                        LaunchConfiguration("rosbag/start_offset/sec"), value_type=float
                    ),
                    "eval/output_tum": LaunchConfiguration("eval/output_tum"),
                    "eval/write_first_frame": ParameterValue(
                        LaunchConfiguration("eval/write_first_frame"), value_type=bool
                    ),
                    "eval/exit_on_end": ParameterValue(
                        LaunchConfiguration("eval/exit_on_end"), value_type=bool
                    ),
                },
            ],
        ),
    ]

    return LaunchDescription(launch_args + nodes)
