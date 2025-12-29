#!/bin/bash
set -e

echo "Setting up workspace..."

# Configure git safe directories
echo "Configuring git safe directories..."
git config --global --add safe.directory /workspace
git config --global --add safe.directory /workspace/src/crane_x7_ros

# Update package lists
echo "Updating apt package lists..."
apt-get update

# Update rosdep
echo "Updating rosdep..."
rosdep update
# Initialize git submodules on correct branch
echo "Initializing git submodules..."
cd /workspace
# Reset any local changes in submodule BEFORE git submodule update
if [ -d src/crane_x7_ros/.git ] || [ -f src/crane_x7_ros/.git ]; then
    echo "Resetting existing submodule changes..."
    cd src/crane_x7_ros
    git reset --hard HEAD 2>/dev/null || true
    git clean -fd 2>/dev/null || true
    cd /workspace
fi
# Initialize and update submodules
git submodule update --init --recursive
# Checkout the correct branch
echo "Checking out humble branch..."
cd /workspace/src/crane_x7_ros
git checkout humble
cd /workspace

# Apply patches for ROS 2 Humble compatibility
echo "Applying compatibility patches..."
cd /workspace/src/crane_x7_ros
if [ -f /workspace/scripts/crane_x7_fixes.patch ]; then
    git apply /workspace/scripts/crane_x7_fixes.patch
    echo "Patches applied successfully"
else
    echo "Warning: patch file not found, skipping patch application"
fi
cd /workspace

# Clean up any misplaced build/install/log directories in src/
echo "Cleaning up misplaced build directories..."
rm -rf /workspace/src/build /workspace/src/install /workspace/src/log

# Install ROS dependencies
echo "Installing ROS dependencies..."
rosdep install -r -y -i --from-paths src --rosdistro ${ROS_DISTRO}

# Create symlinks for header compatibility (.h -> .hpp)
echo "Creating header file symlinks for compatibility..."
# MoveIt headers
if [ -f /opt/ros/humble/include/moveit/move_group_interface/move_group_interface.h ]; then
    ln -sf /opt/ros/humble/include/moveit/move_group_interface/move_group_interface.h \
           /opt/ros/humble/include/moveit/move_group_interface/move_group_interface.hpp
fi
if [ -f /opt/ros/humble/include/moveit/planning_scene_interface/planning_scene_interface.h ]; then
    ln -sf /opt/ros/humble/include/moveit/planning_scene_interface/planning_scene_interface.h \
           /opt/ros/humble/include/moveit/planning_scene_interface/planning_scene_interface.hpp
fi
# cv_bridge headers
if [ -f /opt/ros/humble/include/cv_bridge/cv_bridge/cv_bridge.h ]; then
    ln -sf /opt/ros/humble/include/cv_bridge/cv_bridge/cv_bridge.h \
           /opt/ros/humble/include/cv_bridge/cv_bridge/cv_bridge.hpp
fi
# image_geometry headers
if [ -f /opt/ros/humble/include/image_geometry/image_geometry/pinhole_camera_model.h ]; then
    ln -sf /opt/ros/humble/include/image_geometry/image_geometry/pinhole_camera_model.h \
           /opt/ros/humble/include/image_geometry/image_geometry/pinhole_camera_model.hpp
fi

echo "Setup completed successfully!"
