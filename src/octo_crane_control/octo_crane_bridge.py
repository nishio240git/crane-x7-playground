import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import cv2
import numpy as np

from action_converter import OctoActionConverter

import jax


class OctoCraneBridge(Node):
    def __init__(self):
        super().__init__('octo_crane_bridge')
        self.get_logger().info("OctoCraneBridge started")
        
        self.bridge = CvBridge()
        self.latest_image = None
        
        # パラメータ宣言
        self.declare_parameter('model_name', 'hf://rail-berkeley/octo-small-1.5')
        self.declare_parameter('dataset_stats', 'bridge_dataset')
        self.declare_parameter('language_instruction', 'pick up the cup')
        
        # サブスクライバー
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # パブリッシャー
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/octo/target_pose',
            10
        )
        
        self.load_model()
        
        self.converter = OctoActionConverter()
        initial_pose = [0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
        self.converter.reset(initial_pose)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        
    def load_model(self):
        try:
            from octo.model.octo_model import OctoModel
            
            model_name = self.get_parameter('model_name').value
            self.get_logger().info(f'Loading Octo model: {model_name}')
            
            self.model = OctoModel.load_pretrained(model_name)
            
            self.get_logger().info("Octo model loaded succesfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")    
    
    def image_callback(self,msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg,"rgb8")
        
        resized = cv2.resize(cv_image, (256,256))
        
        normalized = resized / 255.0
        
        self.latest_image = normalized
        self.get_logger().info(f"Image processed: {resized.shape}", throttle_duration_sec=1.0)
    
    def prepare_observation(self):
        if self.latest_image is None:
            return None

        image = np.expand_dims(self.latest_image, axis=0)
        
        image_sequence = np.stack([image, image],axis=1)
        
        return {
            'image_primary': image_sequence,
            'timestep_pad_mask': np.ones((1,2), dtype=bool)
        }
    
    def prepare_task(self):
        """タスク辞書を準備"""
        language_instruction = self.get_parameter('language_instruction').value
        
        return {
            'language_instruction': language_instruction
        }
        
    def infer_action(self, obs, task):
        if not hasattr(self,'rng'):
            self.rng = jax.random.PRNGKey(0)
        
        self.rng, key = jax.random.split(self.rng)
        actions = self.model.sample_actions(
            obs,
            task,
            rng=key,
            unnormalization_statistics=self.get_parameter('dataset_stats').value
        )
        
        action = np.array(actions[0])
        
        return action
    
    def control_loop(self):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet", throttle_duration_sec=5.0)
            return 

        try:
            obs = self.prepare_observation()
            if obs is None:
                return 
            
            task = self.prepare_task()
            
            action = self.infer_action(obs,task)
            
            self.get_logger().info(f"Action: {action}")
            
            # デルタ値を累積
            target_pose = self.converter.accumulate_delta(action)
            
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "crane_x7_mounting_plate_link"
            pose_stamped.pose = target_pose
            
            self.pose_pub.publish(pose_stamped)
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")        
    
    
if __name__=='__main__':
    rclpy.init()
    node = OctoCraneBridge()
    rclpy.spin(node)