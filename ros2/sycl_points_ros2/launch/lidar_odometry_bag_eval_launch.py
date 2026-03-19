from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def declare_params_from_yaml(yaml_path: str, target_node='lidar_odometry_node'):
    launch_args = []
    node_args = {}
    with open(yaml_path, 'r') as f:
        all_params = yaml.safe_load(f)

    for node_name in all_params.keys():
        if node_name == target_node:
            node_params: dict = all_params[node_name]['ros__parameters']
            for name, value in node_params.items():
                if isinstance(value, float):
                    value_str = format(value, 'f')
                else:
                    value_str = str(value)
                launch_args.append(DeclareLaunchArgument(name, default_value=value_str, description=''))
                node_args[name] = LaunchConfiguration(name)
            break
    return launch_args, node_args


def generate_launch_description():
    package_name = 'sycl_points_ros2'
    package_dir = get_package_share_directory(package_name)
    param_yaml = os.path.join(package_dir, 'config', 'lidar_odometry.yaml')
    launch_args, node_args = declare_params_from_yaml(param_yaml, 'lidar_odometry_node')

    launch_args.extend(
        [
            DeclareLaunchArgument('bag_uri', default_value='', description='input rosbag path'),
            DeclareLaunchArgument('point_topic', default_value='/os_cloud_node/points', description='point cloud topic'),
            DeclareLaunchArgument('lidar_frame_id', default_value='os_sensor', description='source point cloud frame'),
            DeclareLaunchArgument('odom_frame_id', default_value='odom', description='odom frame id'),
            DeclareLaunchArgument('base_link_id', default_value='base_link', description='base_link frame id'),
            DeclareLaunchArgument('output_tum', default_value='sycl_lo_odom.tum', description='output tum filepath'),
            DeclareLaunchArgument('bag_start_offset_sec', default_value='0.0', description='bag start offset in seconds'),
            DeclareLaunchArgument('bag_max_frames', default_value='0', description='max frames to process'),
            DeclareLaunchArgument('write_first_frame', default_value='true', choices=['true', 'false'],
                                  description='write first frame pose to tum'),
            DeclareLaunchArgument('exit_on_end', default_value='true', choices=['true', 'false'],
                                  description='shutdown node after evaluation'),
        ]
    )

    nodes = [
        Node(
            package=package_name,
            executable='lidar_odometry_bag_eval_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                node_args,
                {
                    'odom_frame_id': LaunchConfiguration('odom_frame_id'),
                    'base_link_id': LaunchConfiguration('base_link_id'),
                    'bag/uri': LaunchConfiguration('bag_uri'),
                    'bag/topic': LaunchConfiguration('point_topic'),
                    'bag/start_offset_sec': ParameterValue(LaunchConfiguration('bag_start_offset_sec'), value_type=float),
                    'bag/max_frames': ParameterValue(LaunchConfiguration('bag_max_frames'), value_type=int),
                    'eval/output_tum': LaunchConfiguration('output_tum'),
                    'eval/write_first_frame': ParameterValue(LaunchConfiguration('write_first_frame'), value_type=bool),
                    'eval/exit_on_end': ParameterValue(LaunchConfiguration('exit_on_end'), value_type=bool),
                },
            ],
        ),
    ]

    return LaunchDescription(launch_args + nodes)
