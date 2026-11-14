# A single accepted LO/LIO scan and its estimated LiDAR pose at the scan reference time.
# cloud.header.frame_id is the LiDAR frame; lidar_pose.header.frame_id is odom.
# Both headers carry the same scan reference timestamp. No frontend pose Hessian is included.
# When deskewed is false, points were measured at different sensor poses during
# the scan; the single lidar_pose does not motion-compensate those points.
uint64 sequence
bool deskewed
geometry_msgs/PoseStamped lidar_pose
sensor_msgs/PointCloud2 cloud
