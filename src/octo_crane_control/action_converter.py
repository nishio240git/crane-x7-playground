import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R

class OctoActionConverter:
    """
    Octoのデルタアクションを絶対座標に変換するクラス
    """
    
    def __init__(self):
        """初期化：内部状態をNoneで初期化"""
        self.current_position: np.ndarray = None
        self.current_orientation_euler: np.ndarray = None
        self.gripper_state: float = None
    
    def reset(self, initial_pose: np.ndarray) -> None:
        """
        累積計算の基準点を設定
        
        Args:
            initial_pose: [x, y, z, roll, pitch, yaw, gripper]の7次元配列
        
        Raises:
            ValueError: initial_poseの長さが7でない場合
        """
        if len(initial_pose) != 7:
            raise ValueError(f"initial_pose must have 7 elements, got {len(initial_pose)}")
        
        self.current_position = np.array(initial_pose[:3])
        self.current_orientation_euler = np.array(initial_pose[3:6])
        
        self.gripper_state = float(initial_pose[6])
        
        print(f"Reset to position: {self.current_position}")
        print(f"Reset to orientation (euler): {self.current_orientation_euler}")
        print(f"Reset to gripper_state: {self.gripper_state}")
    
    def accumulate_delta(self, delta_action: np.ndarray) -> Pose:
        """
        デルタ値を累積して絶対座標のPoseを返す
        
        Args:
            delta_action: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgrasp]の7次元配列
        
        Returns:
            geometry_msgs/Pose: 累積された絶対座標
        
        Raises:
            RuntimeError: reset()が呼ばれていない場合
            ValueError: delta_actionの長さが7でない場合
        """
        if self.current_position is None:
            raise RuntimeError("reset() must be called before accumulate_delta()")
        
        if len(delta_action) != 7:
            raise ValueError(f"delta_action must have 7 elements, got {len(delta_action)}")

        self.current_position += np.array(delta_action[:3], dtype=np.float64)
        self.current_orientation_euler += np.array(delta_action[3:6], dtype=np.float64)
        self.gripper_state += float(delta_action[6])
        
        self.gripper_state = np.clip(self.gripper_state, 0.0, 1.0)
        
        print(f"Update to position: {self.current_position}")
        print(f"Update to orientation (euler): {self.current_orientation_euler}")
        print(f"Update to gripper_state: {self.gripper_state}")

        # Create Pose
        pose = Pose()
        pose.position = Point(
            x=float(self.current_position[0]),
            y=float(self.current_position[1]),
            z=float(self.current_position[2])
        )
        pose.orientation = self.euler_to_quaternion(self.current_orientation_euler)

        return pose
    
    def get_gripper_command(self) -> float:
        """
        グリッパー状態を実際のグリッパー角度に変換
        
        Returns:
            float: グリッパー角度 [rad] (0.0~0.8)
        
        Raises:
            RuntimeError: reset()が呼ばれていない場合
        """
        if self.gripper_state is None:
            raise RuntimeError("reset() must be called first")
        
        gripper_angle = self.gripper_state * 0.8
        return float(gripper_angle)
    
    @staticmethod
    def euler_to_quaternion(euler: np.ndarray) -> Quaternion:
        """
        オイラー角をQuaternionに変換
        
        Args:
            euler: [roll, pitch, yaw]のオイラー角 [rad]
        
        Returns:
            geometry_msgs/Quaternion
        """
        rot = R.from_euler('xyz', euler)  # Roll→Pitch→Yaw
        quat = rot.as_quat()  # [x, y, z, w]
        
        return Quaternion(
            x = float(quat[0]),
            y = float(quat[1]),
            z = float(quat[2]),
            w = float(quat[3])
        )
    
    @staticmethod
    def quaternion_to_euler(quat: Quaternion) -> np.ndarray:
        """
        Quaternionをオイラー角に変換
        
        Args:
            quat: geometry_msgs/Quaternion
        
        Returns:
            np.ndarray: [roll, pitch, yaw]のオイラー角 [rad]
        """
        quat_array = [quat.x, quat.y, quat.z, quat.w]
        rot = R.from_quat(quat_array)
        euler = rot.as_euler('xyz')  # [roll, pitch, yaw]
        
        return np.array(euler, dtype=np.float64)
      
if __name__ == "__main__":
    # テスト
    initial_pose = [0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]

    action_converter = OctoActionConverter()
    action_converter.reset(initial_pose)

    # 1回目のデルタ値適用
    delta = [0.01, 0.02, 0.0, 0.0, 0.0, 0.1, 0.5]
    pose = action_converter.accumulate_delta(delta)

    print("\n結果:")
    print(f"Position: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")
    print(f"Orientation (quat): ({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, "
          f"{pose.orientation.z:.3f}, {pose.orientation.w:.3f})")
    print(f"Gripper command: {action_converter.get_gripper_command():.3f} rad")
    
    # 2回目のデルタ値適用（累積確認）
    print("\n2回目の累積:")
    delta2 = [0.01, -0.01, 0.01, 0.0, 0.0, 0.05, 0.0]
    pose2 = action_converter.accumulate_delta(delta2)
    print(f"Position: ({pose2.position.x:.3f}, {pose2.position.y:.3f}, {pose2.position.z:.3f})")
    print(f"Orientation (quat): ({pose2.orientation.x:.3f}, {pose2.orientation.y:.3f}, "
          f"{pose2.orientation.z:.3f}, {pose2.orientation.w:.3f})")
    
    # オイラー角への逆変換テスト
    print("\nOrientation変換テスト:")
    euler_back = OctoActionConverter.quaternion_to_euler(pose2.orientation)
    print(f"Euler angles: roll={euler_back[0]:.3f}, pitch={euler_back[1]:.3f}, yaw={euler_back[2]:.3f}") 