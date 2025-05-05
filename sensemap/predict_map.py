import os
import cv2
import numpy as np
import torch
from explore_model.SenseMapNet import DistillMapNet
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry

class SenseMapNetPredictor(Node):
    def __init__(self, model_path):
        super().__init__("SenseMapNetPredictor")
        ckpt = torch.load(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistillMapNet(dim=4).to(self.device)
        weights = {k.split("gen.")[-1]: v for k, v in ckpt["state_dict"].items() if k.startswith("gen.")}
        self.model.load_state_dict(weights)
        self.model.eval()

        self.declare_parameter('robot_id', 0)
        self.robot_id = self.get_parameter('robot_id').value

        # TF2 初始化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_sub_ = self.create_subscription(OccupancyGrid, f"/robot_{self.robot_id}/map", self.global_costmap_callback, 5)
        self.predict_map_pub_ = self.create_publisher(OccupancyGrid, f"/robot_{self.robot_id}/predicted_map", 5)
        self.predict_map_global_pub_ = self.create_publisher(OccupancyGrid, f"/robot_{self.robot_id}/predicted_map_global", 5)
        self.predict_timer = self.create_timer(1.0, self.predict_callback)
        self.frontier_goal_pub_ = self.create_publisher(PoseStamped, f"/robot_{self.robot_id}/way_point", 10)

        self.global_costmap_msg = None
        self.global_costmap_info = None
        self.map_data = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.total_pred_map = None  # 2D numpy array
        self.total_pred_info = None  # OccupancyGrid info
        self.crop_size = 534

    def get_robot_pose_from_tf(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "global_map",
                f"robot_{self.robot_id}/base_link",
                rclpy.time.Time())
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"TF error: {str(e)}")

    def detect_frontiers(self):
        """检测预测地图中的前沿点"""
        if self.total_pred_map is None:
            return []
        
        # 获取地图数据和元信息
        global_data = self.total_pred_map.copy()

        resolution = self.global_costmap_info.resolution
        origin_x = self.global_costmap_info.origin.position.x
        origin_y = self.global_costmap_info.origin.position.y

        map_x = int((self.robot_x - origin_x) / resolution)
        map_y = int((self.robot_y - origin_y) / resolution)
        valid_start_x = max(map_x - self.crop_size//2, 0)
        valid_end_x = min(map_x + self.crop_size//2, self.global_costmap_info.width)
        valid_start_y = max(map_y - self.crop_size//2, 0)
        valid_end_y = min(map_y + self.crop_size//2, self.global_costmap_info.height)

        input_crop = np.full((self.crop_size, self.crop_size), -1, dtype=np.int8)
        input_crop[
                valid_start_y - (map_y - self.crop_size//2):valid_end_y - (map_y - self.crop_size//2),
                valid_start_x - (map_x - self.crop_size//2):valid_end_x - (map_x - self.crop_size//2)
                ] = global_data[valid_start_y:valid_end_y, valid_start_x:valid_end_x]
        new_data = np.full((input_crop.shape[0] + 10, input_crop.shape[1] + 10), -1, dtype=np.int8)
        new_data[5:-5, 5:-5] = input_crop

        # 创建自由空间和未知区域的掩码
        free_mask = (new_data == 0)
        unknown_mask = (new_data == -1)

        # 使用形态学膨胀检测边界
        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(unknown_mask.astype(np.uint8), kernel)

        cv2.imwrite("/home/azusa/SenseLabRobo/SenseServer/dilated_unknown.png", dilated_unknown)

        # 计算前沿点（自由区域与未知区域相邻）
        frontiers = np.logical_and(free_mask, dilated_unknown)
        frontier_coords = np.argwhere(frontiers)

        # 转换为世界坐标系
        world_points = []
        for y, x in frontier_coords:
            wx = origin_x + (x + 0.5) * resolution  # 网格中心坐标
            wy = origin_y + (y + 0.5) * resolution
            world_points.append([wx, wy])
        
        return world_points
    
    def cluster_frontiers(self, points):
        """使用DBSCAN算法对前沿点进行聚类"""
        if len(points) < 5:  # 最少需要5个点形成有效聚类
            return []
        
        dbscan = DBSCAN(eps=1.0, min_samples=5)  # 参数根据分辨率调整（1米范围和最少5个点）
        clusters = dbscan.fit_predict(points)
        
        # 提取有效聚类
        unique_labels = np.unique(clusters)
        clustered_points = []
        for label in unique_labels:
            if label == -1:
                continue  # 跳过噪声点
            clustered_points.append(points[clusters == label])
        
        return clustered_points
    
    def select_best_frontier(self, clusters):
       """选择最优前沿区域（最近且最大的集群）"""
       if not clusters:
           return None
       
       current_pos = np.array([self.robot_x, self.robot_y])
       best_score = -np.inf
       best_center = None
       
       for cluster in clusters:
           # 计算集群中心
           center = np.mean(cluster, axis=0)
           # 计算到机器人的距离
           distance = np.linalg.norm(center - current_pos)
           # 计算集群规模
           size = len(cluster)
           # 综合评分（越大越好）
           score = abs(distance)  # 避免除以零
           
           if score > best_score:
               best_score = score
               best_center = center
       
       return best_center

    def predict(self, img):
        self.get_robot_pose_from_tf()
        img = torch.tensor(img, dtype=torch.float32).to(self.device) / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            pred, _, _, _, _ = self.model(img)
        return pred.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    
    def global_costmap_callback(self, msg: OccupancyGrid):
        self.global_costmap_msg = msg
        self.global_costmap_info = msg.info

    def expand_global_map(self, pred_origin_x, pred_origin_y, crop_size, resolution):
        if self.total_pred_map is None:
            self.total_pred_info = OccupancyGrid().info
            self.total_pred_info.resolution = resolution
            self.total_pred_info.width = crop_size
            self.total_pred_info.height = crop_size
            self.total_pred_info.origin.position.x = pred_origin_x
            self.total_pred_info.origin.position.y = pred_origin_y
            self.total_pred_map = np.full((crop_size, crop_size), -1, dtype=np.int8)
            return

        # 计算当前全局地图的边界
        curr_origin_x = self.total_pred_info.origin.position.x
        curr_origin_y = self.total_pred_info.origin.position.y
        curr_width = self.total_pred_info.width
        curr_height = self.total_pred_info.height
        curr_max_x = curr_origin_x + curr_width * resolution
        curr_max_y = curr_origin_y + curr_height * resolution

        # 预测区域的边界
        pred_min_x = pred_origin_x
        pred_min_y = pred_origin_y
        pred_max_x = pred_origin_x + crop_size * resolution
        pred_max_y = pred_origin_y + crop_size * resolution

        # 计算新全局地图的边界
        new_min_x = min(curr_origin_x, pred_min_x)
        new_min_y = min(curr_origin_y, pred_min_y)
        new_max_x = max(curr_max_x, pred_max_x)
        new_max_y = max(curr_max_y, pred_max_y)

        new_width = int(np.ceil((new_max_x - new_min_x) / resolution))
        new_height = int(np.ceil((new_max_y - new_min_y) / resolution))

        # 调整原点以确保整数倍分辨率
        new_origin_x = new_min_x
        new_origin_y = new_min_y

        # 创建新地图并复制旧数据
        new_map = np.full((new_height, new_width), -1, dtype=np.int8)
        offset_x = int(round((curr_origin_x - new_origin_x) / resolution))
        offset_y = int(round((curr_origin_y - new_origin_y) / resolution))
        new_map[offset_y:offset_y+curr_height, offset_x:offset_x+curr_width] = self.total_pred_map

        # 更新全局地图信息
        self.total_pred_info.origin.position.x = new_origin_x
        self.total_pred_info.origin.position.y = new_origin_y
        self.total_pred_info.width = new_width
        self.total_pred_info.height = new_height
        self.total_pred_map = new_map

    def update_global_map(self, pred_data, pred_origin_x, pred_origin_y, crop_size, resolution):
        total_origin_x = self.total_pred_info.origin.position.x
        total_origin_y = self.total_pred_info.origin.position.y
        total_width = self.total_pred_info.width
        total_height = self.total_pred_info.height

        # 计算预测区域在全局地图中的索引范围
        start_x = int(round((pred_origin_x - total_origin_x) / resolution))
        start_y = int(round((pred_origin_y - total_origin_y) / resolution))
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # 处理边界情况
        valid_start_x = max(start_x, 0)
        valid_end_x = min(end_x, total_width)
        valid_start_y = max(start_y, 0)
        valid_end_y = min(end_y, total_height)

        # 计算对应的数据切片
        local_start_x = valid_start_x - start_x
        local_start_y = valid_start_y - start_y
        local_end_x = local_start_x + (valid_end_x - valid_start_x)
        local_end_y = local_start_y + (valid_end_y - valid_start_y)

        if local_start_x < local_end_x and local_start_y < local_end_y:
            self.total_pred_map[valid_start_y:valid_end_y, valid_start_x:valid_end_x] = (0.75 * self.total_pred_map[valid_start_y:valid_end_y, valid_start_x:valid_end_x]).astype(np.int8)
            self.total_pred_map[valid_start_y:valid_end_y, valid_start_x:valid_end_x] += (0.25 * pred_data[local_start_y:local_end_y, local_start_x:local_end_x]).astype(np.int8)

    def predict_callback(self):
        if self.global_costmap_msg is None or self.global_costmap_info is None:
            return

        resolution = self.global_costmap_info.resolution
        origin_x = self.global_costmap_info.origin.position.x
        origin_y = self.global_costmap_info.origin.position.y

        # 获取机器人在地图中的位置
        map_x = int((self.robot_x - origin_x) / resolution)
        map_y = int((self.robot_y - origin_y) / resolution)

        # 创建输入裁剪区域
        input_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        input_crop[:, :, 1] = 255  # 初始化为未知

        # 计算有效区域
        valid_start_x = max(map_x - self.crop_size//2, 0)
        valid_end_x = min(map_x + self.crop_size//2, self.global_costmap_info.width)
        valid_start_y = max(map_y - self.crop_size//2, 0)
        valid_end_y = min(map_y + self.crop_size//2, self.global_costmap_info.height)

        # 填充输入数据
        if valid_end_x > valid_start_x and valid_end_y > valid_start_y:
            self.map_data = np.array(self.global_costmap_msg.data).reshape((self.global_costmap_info.height, self.global_costmap_info.width))
            input_crop[
                valid_start_y - (map_y - self.crop_size//2):valid_end_y - (map_y - self.crop_size//2),
                valid_start_x - (map_x - self.crop_size//2):valid_end_x - (map_x - self.crop_size//2), 
                0] = (self.map_data[valid_start_y:valid_end_y, valid_start_x:valid_end_x] == 100) * 255  # 障碍
            input_crop[
                valid_start_y - (map_y - self.crop_size//2):valid_end_y - (map_y - self.crop_size//2),
                valid_start_x - (map_x - self.crop_size//2):valid_end_x - (map_x - self.crop_size//2), 
                1] = (self.map_data[valid_start_y:valid_end_y, valid_start_x:valid_end_x] == -1) * 255  # 未知区域
            input_crop[
                valid_start_y - (map_y - self.crop_size//2):valid_end_y - (map_y - self.crop_size//2),
                valid_start_x - (map_x - self.crop_size//2):valid_end_x - (map_x - self.crop_size//2), 
                2] = (self.map_data[valid_start_y:valid_end_y, valid_start_x:valid_end_x] == 0) * 255   # 自由区域

        # 预处理并预测
        img = cv2.resize(input_crop, (256, 256))
        pred = self.predict(img)
        pred = cv2.resize(pred, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        pub_img = cv2.threshold(pred, 0.5, 100, cv2.THRESH_BINARY)[1].astype(np.int8)

        # 计算预测区域的全局原点
        pred_origin_x = self.robot_x - (self.crop_size // 2) * resolution
        pred_origin_y = self.robot_y - (self.crop_size // 2) * resolution

        # 扩展并更新全局地图
        self.expand_global_map(pred_origin_x, pred_origin_y, self.crop_size, resolution)
        self.update_global_map(pub_img, pred_origin_x, pred_origin_y, self.crop_size, resolution)

        # 发布全局预测地图
        pred_msg = OccupancyGrid()
        pred_msg.header.stamp = self.get_clock().now().to_msg()
        pred_msg.header.frame_id = "global_map"
        pred_msg.info = self.total_pred_info
        pred_msg.data = self.total_pred_map.flatten().astype(np.int8).tolist()
        self.predict_map_global_pub_.publish(pred_msg)

        # 发布局部预测地图
        local_pred_msg = OccupancyGrid()
        local_pred_msg.header.stamp = self.get_clock().now().to_msg()
        local_pred_msg.header.frame_id = "global_map"
        local_pred_msg.info.height = self.crop_size
        local_pred_msg.info.width = self.crop_size
        local_pred_msg.info.resolution = resolution
        local_pred_msg.info.origin.position.x = self.robot_x - (self.crop_size // 2) * resolution
        local_pred_msg.info.origin.position.y = self.robot_y - (self.crop_size // 2) * resolution
        local_pred_msg.data = pub_img.flatten().astype(np.int8).tolist()
        self.predict_map_pub_.publish(local_pred_msg)

        # raw_points = self.detect_frontiers()
        # if len(raw_points) > 0:
        #     points_array = np.array(raw_points)
        #     clusters = self.cluster_frontiers(points_array)
        #     best_goal = self.select_best_frontier(clusters)
        #     if best_goal is not None:
        #         goal_msg = PoseStamped()
        #         goal_msg.header.stamp = self.get_clock().now().to_msg()
        #         goal_msg.header.frame_id = "global_map"
        #         goal_msg.pose.position.x = best_goal[0]
        #         goal_msg.pose.position.y = best_goal[1]
        #         self.frontier_goal_pub_.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    model_path = os.path.join(os.path.dirname(__file__), "config/model-epoch-label.ckpt")
    node = SenseMapNetPredictor(model_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()