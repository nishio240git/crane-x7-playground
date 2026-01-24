from rclpy.node import Node
import rclpy
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped

class CartesianExecutor(Node):
    def __init__(self,move_group=None):
        super().__init__('cartesian_executor')
        # declare parameter
        
        if move_group is None:
            self.move_group = MoveItPy(node=self)
        else:
            self.move_group = move_group
        
        self.target_pose: PoseStamped = None
        self.ik_success_count = 0
        self.ik_fail_count = 0
        
        if self.move_group is None:
            self.init_move_group()
            self.subscription = self.create_subscription(
            PoseStamped,
            'target_pose',  # トピック名
            self.target_callback,
            10
        )
        
            self.timer = self.create_timer(0.1, self.execute_if_ready)

    def init_move_group(self):
        self.arm = self.move_group.get_planning_component("arm")
        
        self.arm.set_start_state_to_current_state()
    
    def target_callback(self, msg: PoseStamped):
        self.target_pose = msg
        self.get_logger().info(f"Received target pose: [{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]")
    
    def execute_if_ready(self):
        if self.target_pose is not None:
            self.execute_target()
    
    def execute_target(self) -> bool:
        if self.target_pose is None:
            return False
        
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=self.target_pose, pose_link="crane_x7_gripper_base_link")
    
        plan_result = self.arm.plan()
        if plan_result:
            self.arm.execute()
            
            self.ik_success_count += 1
            self.get_logger().info(f"Execution succeeded. Success: {self.ik_success_count}, Failed: {self.ik_fail_count}")
            self.target_pose = None
            return True
        else:
            self.ik_fail_count += 1
            self.get_logger().warn("Planning failed")
            return False
    
    