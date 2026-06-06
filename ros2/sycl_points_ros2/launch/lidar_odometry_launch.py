from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.actions import TimerAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
import yaml


def declare_params_from_yaml(yaml_path: str, target_node="lidar_odometry_node"):
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
    node_name = "lidar_odometry_node"
    package_dir = get_package_share_directory(package_name)
    param_yaml = os.path.join(package_dir, "config", "lidar_odometry.yaml")
    launch_args, node_args = declare_params_from_yaml(param_yaml, node_name)
    launch_args.extend(
        [
            DeclareLaunchArgument(
                "rosbag/play",
                default_value="false",
                description="play rosbag or not",
            ),
            DeclareLaunchArgument(
                "rosbag/uri",
                default_value="",
                description="rosbag path",
            ),
            DeclareLaunchArgument(
                "rosbag/start_offset/sec",
                default_value="0",
                description="rosbag start offset in seconds",
            ),
            DeclareLaunchArgument(
                "rosbag/exclude_topics",
                default_value="",
                description=(
                    "Comma-separated topics to exclude from rosbag playback "
                    "(e.g. '/tf_static' when the bag's static TF conflicts with the "
                    "node's odom->base_link->lidar tree). Empty = exclude nothing. "
                    "Does NOT affect odometry results (the pipeline never consumes TF); "
                    "visualization only."
                ),
            ),
        ]
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    def launch_setup(context, *args, **kwargs):
        # Parse the comma-separated exclude list at runtime.  Omit the parameter
        # entirely when empty: an empty array param has an ambiguous type and would
        # otherwise error, and omitting it keeps the default "exclude nothing".
        exclude_raw = LaunchConfiguration("rosbag/exclude_topics").perform(context)
        exclude_topics = [t.strip() for t in exclude_raw.split(",") if t.strip()]

        # Resolve substitutions to concrete Python types here so the rosbag2 Player
        # (which is strict about parameter types) receives int/bool/str rather than
        # raw substitution strings.
        try:
            start_offset_sec = int(
                LaunchConfiguration("rosbag/start_offset/sec").perform(context)
            )
        except ValueError:
            start_offset_sec = 0
        sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"

        player_params = {
            "play.read_ahead_queue_size": 1000,
            "play.node_prefix": "",
            "play.rate": 1.0,
            "play.loop": False,
            "play.start_paused": False,
            "play.start_offset.sec": start_offset_sec,
            "storage.uri": LaunchConfiguration("rosbag/uri").perform(context),
            "storage.storage_config_uri": "",
            "use_sim_time": sim_time,
        }
        if exclude_topics:
            player_params["play.exclude_topics_to_filter"] = exclude_topics

        container = TimerAction(
            period=1.0,
            actions=[
                ComposableNodeContainer(
                    name="sycl_points_container",
                    namespace="",
                    package="rclcpp_components",
                    # executable='component_container',  # SingleThreadedExecutor
                    executable="component_container_mt",  # MultiThreadedExecutor
                    output="screen",
                    emulate_tty=True,
                    composable_node_descriptions=[
                        ComposableNode(
                            package="tf2_ros",
                            plugin="tf2_ros::StaticTransformBroadcasterNode",
                            name="static_transform_broadcaster",
                            parameters=[
                                {
                                    "translation.x": LaunchConfiguration(
                                        "T_base_link_to_lidar/x"
                                    ),
                                    "translation.y": LaunchConfiguration(
                                        "T_base_link_to_lidar/y"
                                    ),
                                    "translation.z": LaunchConfiguration(
                                        "T_base_link_to_lidar/z"
                                    ),
                                    "rotation.x": LaunchConfiguration(
                                        "T_base_link_to_lidar/qx"
                                    ),
                                    "rotation.y": LaunchConfiguration(
                                        "T_base_link_to_lidar/qy"
                                    ),
                                    "rotation.z": LaunchConfiguration(
                                        "T_base_link_to_lidar/qz"
                                    ),
                                    "rotation.w": LaunchConfiguration(
                                        "T_base_link_to_lidar/qw"
                                    ),
                                    "frame_id": LaunchConfiguration("base_link_id"),
                                    "child_frame_id": LaunchConfiguration(
                                        "lidar_frame_id"
                                    ),
                                    "use_sim_time": use_sim_time,
                                },
                            ],
                        ),
                        # rosbag2_transport must load before sycl_points
                        ComposableNode(
                            package="rosbag2_transport",
                            plugin="rosbag2_transport::Player",
                            name="player",
                            parameters=[player_params],
                            condition=IfCondition(LaunchConfiguration("rosbag/play")),
                            extra_arguments=[{"use_intra_process_comms": True}],
                        ),
                        ComposableNode(
                            package=package_name,
                            plugin="sycl_points::ros2::LiDAROdometryNode",
                            name=package_name,
                            parameters=[
                                node_args,
                                {"use_sim_time": use_sim_time},
                            ],
                            extra_arguments=[{"use_intra_process_comms": True}],
                        ),
                    ],
                ),
            ],
        )
        return [container]

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(package_dir, "rviz2", "rviz2.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz2")),
    )

    return LaunchDescription(launch_args + [rviz, OpaqueFunction(function=launch_setup)])
