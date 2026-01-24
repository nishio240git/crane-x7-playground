import pytest
from unittest.mock import Mock, MagicMock, patch, call
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import rclpy

import sys
sys.modules['moveit'] = MagicMock()
sys.modules['moveit.planning'] = MagicMock()

from octo_crane_control.cartesian_executor import CartesianExecutor

class TestCartesianExecutor:
    
    @pytest.fixture
    def mock_move_group(self):
        mock = Mock()
        mock_arm = Mock()
        mock.get_planning_component.return_value = mock_arm
        return mock, mock_arm
    
    @pytest.fixture
    def executor_with_mock(self, mock_move_group):
        mock_moveit, mock_arm = mock_move_group
        
        if not rclpy.ok():
            rclpy.init()
                
        executor = CartesianExecutor(move_group=mock_moveit)
        executor.arm = mock_arm
        
        return executor, mock_arm

    def test_init_with_mock_move_group(self, mock_move_group):
        """move_groupを渡した場合の初期化テスト"""
        # Arrange
        mock_moveit, mock_arm = mock_move_group
        
        if not rclpy.ok():
            rclpy.init()
        
        # Act
        executor = CartesianExecutor(move_group=mock_moveit)
        # Assert
        assert executor.move_group == mock_moveit
        assert executor.target_pose is None
        assert executor.ik_success_count == 0
        assert executor.ik_fail_count == 0
        
        

    def test_target_callback_start_pose(self, executor_with_mock):
        # Arrange
        executor, _ = executor_with_mock
        msg = PoseStamped()
        msg.pose.position = Point(x=0.3, y=0.2, z=0.3)
        msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        # Act
        executor.target_callback(msg)
        
        # Assert
        assert executor.target_pose == msg
        assert executor.target_pose.pose.position.x == 0.3
        
    def test_execute_if_ready_does_nothing_when_no_target(self, executor_with_mock):
        """target_poseがNoneの時は何もしないことを確認"""
        # Arrange
        executor, mock_arm = executor_with_mock
        executor.target_pose = None
        
        # Act
        executor.execute_if_ready()
        
        # Assert
        mock_arm.set_start_state_to_current_state.assert_not_called()
    
    def test_execute_if_ready_executes_when_target_exists(self, executor_with_mock):
        """target_poseがある時はexecute_targetが呼ばれることを確認"""
        # Arrange
        executor, mock_arm = executor_with_mock
        msg = PoseStamped()
        msg.pose.position = Point(x=0.3, y=0.2, z=0.3)
        executor.target_pose = msg
        
        mock_arm.plan.return_value = True  # プランニング成功
        
        # Act
        executor.execute_if_ready()
        
        # Assert
        mock_arm.set_start_state_to_current_state.assert_called()
    
    def test_execute_target_returns_false_when_no_target(self, executor_with_mock):
        """target_poseがNoneの時はFalseを返すことを確認"""
        # Arrange
        executor, _ = executor_with_mock
        executor.target_pose = None
        
        # Act
        result = executor.execute_target()
        
        # Assert
        assert result is False
    
    def test_execute_target_succeeds_with_valid_plan(self, executor_with_mock):
        """プランニング成功時の挙動を確認"""
        # Arrange
        executor, mock_arm = executor_with_mock
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position = Point(x=0.3, y=0.2, z=0.3)
        msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        executor.target_pose = msg
        
        mock_arm.plan.return_value = True  # プランニング成功
        
        # Act
        result = executor.execute_target()
        
        # Assert
        assert result is True
        assert executor.ik_success_count == 1
        assert executor.ik_fail_count == 0
        assert executor.target_pose is None  # クリアされる
        mock_arm.set_start_state_to_current_state.assert_called_once()
        mock_arm.set_goal_state.assert_called_once_with(
            pose_stamped_msg=msg,
            pose_link="crane_x7_gripper_base_link"
        )
        mock_arm.plan.assert_called_once()
        mock_arm.execute.assert_called_once()
    
    def test_execute_target_fails_with_invalid_plan(self, executor_with_mock):
        """プランニング失敗時の挙動を確認"""
        # Arrange
        executor, mock_arm = executor_with_mock
        msg = PoseStamped()
        msg.pose.position = Point(x=0.3, y=0.2, z=0.3)
        executor.target_pose = msg
        
        mock_arm.plan.return_value = False  # プランニング失敗
        
        # Act
        result = executor.execute_target()
        
        # Assert
        assert result is False
        assert executor.ik_success_count == 0
        assert executor.ik_fail_count == 1
        assert executor.target_pose == msg  # クリアされない
        mock_arm.execute.assert_not_called()  # executeは呼ばれない
        
    def test_execute_target_increments_counters_correctly(self, executor_with_mock):
        """複数回の実行でカウンタが正しく増加することを確認"""
        # Arrange
        executor, mock_arm = executor_with_mock
        msg = PoseStamped()
        msg.pose.position = Point(x=0.3, y=0.2, z=0.3)
        
        # Act & Assert: 成功2回
        executor.target_pose = msg
        mock_arm.plan.return_value = True
        executor.execute_target()
        
        executor.target_pose = msg
        mock_arm.plan.return_value = True
        executor.execute_target()
        
        assert executor.ik_success_count == 2
        assert executor.ik_fail_count == 0
        
        # Act & Assert: 失敗1回
        executor.target_pose = msg
        mock_arm.plan.return_value = False
        executor.execute_target()
        
        assert executor.ik_success_count == 2
        assert executor.ik_fail_count == 1