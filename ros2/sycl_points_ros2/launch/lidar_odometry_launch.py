from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.actions import TimerAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
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
                launch_args.append(
                    DeclareLaunchArgument(name, default_value=value_str, description='')
                )
                node_args[name] = LaunchConfiguration(name)
            break
    return launch_args, node_args


def generate_launch_description():
    package_name = 'sycl_points_ros2'
    node_name = 'lidar_odometry_node'
    package_dir = get_package_share_directory(package_name)
    param_yaml = os.path.join(package_dir, 'config', 'lidar_odometry.yaml')
    launch_args, node_args = declare_params_from_yaml(param_yaml, node_name)
    launch_args.extend(
        [
            DeclareLaunchArgument(
                'point_topic',
                default_value='/os_cloud_node/points',
                description='source point cloud topic',
            ),
            DeclareLaunchArgument(
                'lidar_frame_id',
                default_value='os_sensor',
                description='source point cloud frame id',
            ),
            DeclareLaunchArgument(
                'rviz2',
                default_value='true',
                choices=['true', 'false'],
                description='launch with rviz2',
            ),
            DeclareLaunchArgument(
                'odom_frame_id',
                default_value='odom',
                description='odom frame id',
            ),
            DeclareLaunchArgument(
                'base_link_id',
                default_value='base_link',
                description='base_link frame id',
            ),
            DeclareLaunchArgument(
                'rosbag/play',
                default_value='false',
                description='play rosbag or not',
            ),
            DeclareLaunchArgument(
                'rosbag/uri',
                default_value='',
                description='rosbag path',
            ),
        ]
    )

    nodes = [
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(package_dir, 'rviz2', 'rviz2.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz2')),
        ),
        TimerAction(
            period=1.0,
            actions=[
                ComposableNodeContainer(
                    name='sycl_points_container',
                    namespace='',
                    package='rclcpp_components',
                    # executable='component_container',  # SingleThreadedExecutor
                    executable='component_container_mt',  # MultiThreadedExecutor
                    output='screen',
                    emulate_tty=True,
                    composable_node_descriptions=[
                        ComposableNode(
                            package='tf2_ros',
                            plugin='tf2_ros::StaticTransformBroadcasterNode',
                            name='static_transform_broadcaster',
                            parameters=[
                                {
                                    'translation.x': LaunchConfiguration('T_base_link_to_lidar/x'),
                                    'translation.y': LaunchConfiguration('T_base_link_to_lidar/y'),
                                    'translation.z': LaunchConfiguration('T_base_link_to_lidar/z'),
                                    'rotation.x': LaunchConfiguration('T_base_link_to_lidar/qx'),
                                    'rotation.y': LaunchConfiguration('T_base_link_to_lidar/qy'),
                                    'rotation.z': LaunchConfiguration('T_base_link_to_lidar/qz'),
                                    'rotation.w': LaunchConfiguration('T_base_link_to_lidar/qw'),
                                    'frame_id': LaunchConfiguration('base_link_id'),
                                    'child_frame_id': LaunchConfiguration('lidar_frame_id'),
                                },
                            ]
                        ),
                        ComposableNode(
                            package=package_name,
                            plugin='sycl_points::ros2::LiDAROdometryNode',
                            name=package_name,
                            parameters=[
                                node_args,
                                {
                                    'odom_frame_id': LaunchConfiguration(
                                        'odom_frame_id'
                                    ),
                                    'base_link_id': LaunchConfiguration('base_link_id'),
                                },
                            ],
                            remappings=[
                                ('points', LaunchConfiguration('point_topic')),
                            ],
                            extra_arguments=[{'use_intra_process_comms': True}],
                        ),
                        ComposableNode(
                            package='rosbag2_transport',
                            plugin='rosbag2_transport::Player',
                            name='player',
                            parameters=[
                                {
                                    'play.read_ahead_queue_size': 1000,
                                    'play.node_prefix': '',
                                    'play.rate': 1.0,
                                    'play.loop': False,
                                    'play.start_paused': False,
                                    'storage.uri': LaunchConfiguration('rosbag/uri'),
                                    'storage.storage_config_uri': '',
                                }
                            ],
                            condition=IfCondition(LaunchConfiguration('rosbag/play')),
                            extra_arguments=[{'use_intra_process_comms': True}],
                        ),
                    ],
                ),
            ],
        ),
    ]

    return LaunchDescription(launch_args + nodes)
