import pytest
import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from octo_crane_control.action_converter import OctoActionConverter

class TestOctoActionConverter:
    
    def test_reset_sets_initial_state(self):
        """reset()が初期状態を正しく設定することを確認"""
        ## Arrange
        converter = OctoActionConverter()
        initial_pose = [0.3,0.2,0.3,0.0,0.0,0.0,0.5]
        
        # Act
        converter.reset(initial_pose)
        
        # Assert
        np.testing.assert_array_equal(converter.current_position, np.array([0.3, 0.2, 0.3]))
        np.testing.assert_array_equal(converter.current_orientation_euler,np.array([0.0, 0.0, 0.0]))
        assert converter.gripper_state == 0.5
    
    def test_reset_raises_error_for_invalid_length(self):
        """reset()が不正な長さの配列でValueErrorを発生させることを確認"""
        # Arrange
        converter = OctoActionConverter()
        invalid_pose = [0.3, 0.2, 0.3]
        
        # Act & Assert
        with pytest.raises(ValueError, match="initial_pose must have 7 elements"):
            converter.reset(invalid_pose)
    
    def test_accumurate_delta_updates_position(self):
        """reset()なしでaccumulate_delta()を呼ぶとRuntimeErrorを発生させることを確認"""
        # Arrange
        converter = OctoActionConverter()
        initial_pose = [0.3,0.2,0.3,0.0,0.0,0.0,0.5]
        converter.reset(initial_pose)
        delta = [0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0]
        
        # Act
        pose = converter.accumulate_delta(delta)
        
        # Asset
        assert pose.position.x == pytest.approx(0.31, abs=1e-6)
        assert pose.position.y == pytest.approx(0.22, abs=1e-6)
        assert pose.position.z == pytest.approx(0.33, abs=1e-6)
        
    def test_accumurate_delta_raises_error_without_reset(self):
        """accumulate_delta()が不正な長さの配列でValueErrorを発生させることを確認"""
        # Arrange 
        converter = OctoActionConverter()
        delta = [0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0]
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="reset\\(\\) must be called before accumulate_delta"):
            converter.accumulate_delta(delta)
    
    def test_accumulate_delta_raises_error_for_invalid_length(self):
        """accumulate_delta()が不正な長さの配列でValueErrorを発生させることを確認"""
        # Arrange
        converter = OctoActionConverter()
        converter.reset([0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
        invalid_delta = [0.01, 0.02, 0.03]  # 長さ3（正しくは7）
        
        # Act & Assert
        with pytest.raises(ValueError, match="delta_action must have 7 elements"):
            converter.accumulate_delta(invalid_delta)
    
    def test_gripper_state_clips_to_valid_range(self):
        """gripper_stateが0.0-1.0の範囲にクリップされることを確認"""
        # Arrange
        converter = OctoActionConverter()
        converter.reset([0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])
        
        # Act: gripper_stateを1.0以上に増やす試み
        converter.accumulate_delta([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
        
        # Assert
        assert converter.gripper_state == 1.0
        
        # Act: gripper_stateを0.0以下に減らす試み
        converter.accumulate_delta([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
        
        # Assert
        assert converter.gripper_state == 0.0
    
    def test_get_gripper_command_converts_correctly(self):
        """get_gripper_command()が正しくグリッパー角度に変換することを確認"""
        # Arrange
        converter = OctoActionConverter()
        converter.reset([0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])
        
        # Act
        gripper_angle = converter.get_gripper_command()
        
        # Assert
        assert gripper_angle == pytest.approx(0.4, abs=1e-6)  # 0.5 * 0.8 = 0.4
    
    def test_get_gripper_command_raises_error_without_reset(self):
        """reset()なしでget_gripper_command()を呼ぶとRuntimeErrorを発生させることを確認"""
        # Arrange
        converter = OctoActionConverter()
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="reset\\(\\) must be called first"):
            converter.get_gripper_command()
        
    def test_euler_to_quaternion_identity_rotation(self):
        """euler_to_quaternion()が単位回転を正しく変換することを確認"""
        # Arrange
        euler = np.array([0.0, 0.0, 0.0])
        
        # Act
        quat = OctoActionConverter.euler_to_quaternion(euler)
        
        # Assert
        assert quat.x == pytest.approx(0.0, abs=1e-6)
        assert quat.y == pytest.approx(0.0, abs=1e-6)
        assert quat.z == pytest.approx(0.0, abs=1e-6)
        assert quat.w == pytest.approx(1.0, abs=1e-6)
    
    def test_euler_to_quaternion_90deg_roll(self):
        """euler_to_quaternion()が90度ロールを正しく変換することを確認"""
        # Arrange
        euler = np.array([np.pi/2, 0.0, 0.0])  # 90度ロール
        
        # Act
        quat = OctoActionConverter.euler_to_quaternion(euler)
        
        # Assert
        assert quat.x == pytest.approx(0.7071068, abs=1e-6)
        assert quat.y == pytest.approx(0.0, abs=1e-6)
        assert quat.z == pytest.approx(0.0, abs=1e-6)
        assert quat.w == pytest.approx(0.7071068, abs=1e-6)
    
    def test_quaternion_to_euler_identity_rotation(self):
        """quaternion_to_euler()が単位回転を正しく変換することを確認"""
        # Arrange
        quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # Act
        euler = OctoActionConverter.quaternion_to_euler(quat)
        
        # Assert
        np.testing.assert_array_almost_equal(euler, [0.0, 0.0, 0.0], decimal=6)
    
    def test_euler_quaternion_round_trip(self):
        """オイラー角→クォータニオン→オイラー角の往復変換が正しいことを確認"""
        # Arrange
        original_euler = np.array([0.1,0.2,0.3])
        
        # Act
        quat = OctoActionConverter.euler_to_quaternion(original_euler)
        result_euler = OctoActionConverter.quaternion_to_euler(quat)
        
        # Assert
        np.testing.assert_array_almost_equal(result_euler,original_euler,decimal=6)
        
    def test_accumulate_delta_updates_orientation(self):
        """accumulate_delta()が姿勢を正しく累積することを確認"""
        # Arrange
        converter = OctoActionConverter()
        converter.reset([0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
        delta = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0]
        
        # Act
        pose = converter.accumulate_delta(delta)
        
        # Assert: Quaternionから逆変換して確認
        euler_back = OctoActionConverter.quaternion_to_euler(pose.orientation)
        np.testing.assert_array_almost_equal(euler_back, [0.1, 0.2, 0.3], decimal=6)
    