#!/bin/bash
set -e

echo "Setting up workspace..."

# Configure git safe directories
echo "Configuring git safe directories..."
git config --global --add safe.directory /workspace
git config --global --add safe.directory /workspace/src/crane_x7_ros
git config --global --add safe.directory /workspace/src/octo
git config --global --add safe.directory /workspace/src/rt_manipulators_cpp

# Update package lists
echo "Updating apt package lists..."
sudo mkdir -p /var/lib/apt/lists/partial
sudo apt-get update

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
echo "Checking out jazzy branch..."
cd /workspace/src/crane_x7_ros
git checkout jazzy
cd /workspace/src/crane_x7_description
git checkout jazzy
cd /workspace

# Apply patches for ROS 2 Humble compatibility (only if needed)
echo "Checking if compatibility patches are needed..."
cd /workspace/src/crane_x7_ros
if [ -f /workspace/scripts/crane_x7_fixes.patch ]; then
    # Try to apply patch, but don't fail if already applied
    git apply /workspace/scripts/crane_x7_fixes.patch 2>/dev/null && echo "Patches applied successfully" || echo "Patches already applied or not needed for this branch"
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

# Install Octo package (in Python 3.10 venv)
echo "Installing Octo package in virtual environment..."
cd /workspace/src/octo
/opt/octo_env/bin/pip install -e .
/opt/octo_env/bin/pip install git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972
cd /workspace

# Create symlinks for header compatibility (.h -> .hpp)
echo "Creating header file symlinks for compatibility..."
# MoveIt headers
if [ -f /opt/ros/jazzy/include/moveit/move_group_interface/move_group_interface.h ]; then
    sudo ln -sf /opt/ros/jazzy/include/moveit/move_group_interface/move_group_interface.h \
           /opt/ros/jazzy/include/moveit/move_group_interface/move_group_interface.hpp
fi
if [ -f /opt/ros/jazzy/include/moveit/planning_scene_interface/planning_scene_interface.h ]; then
    sudo ln -sf /opt/ros/jazzy/include/moveit/planning_scene_interface/planning_scene_interface.h \
           /opt/ros/jazzy/include/moveit/planning_scene_interface/planning_scene_interface.hpp
fi
# cv_bridge headers
if [ -f /opt/ros/jazzy/include/cv_bridge/cv_bridge/cv_bridge.h ]; then
    sudo ln -sf /opt/ros/jazzy/include/cv_bridge/cv_bridge/cv_bridge.h \
           /opt/ros/jazzy/include/cv_bridge/cv_bridge/cv_bridge.hpp
fi
# image_geometry headers
if [ -f /opt/ros/jazzy/include/image_geometry/image_geometry/pinhole_camera_model.h ]; then
    sudo ln -sf /opt/ros/jazzy/include/image_geometry/image_geometry/pinhole_camera_model.h \
           /opt/ros/jazzy/include/image_geometry/image_geometry/pinhole_camera_model.hpp
fi

echo "Setup completed successfully!"
