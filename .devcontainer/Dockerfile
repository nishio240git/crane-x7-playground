FROM osrf/ros:humble-desktop-full

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-gazebo-ros2-control \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-moveit \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Octoの依存関係をインストール
RUN pip3 install --upgrade pip
# JAXとjaxlibを先にインストール（バージョンを揃える）
RUN pip3 install jax==0.4.20 jaxlib==0.4.20
RUN pip3 install \
    gym>=0.26 \
    numpy==1.24.3 \
    ml_dtypes==0.2.0 \
    chex==0.1.85 \
    optax==0.1.5 \
    tensorflow_probability==0.23.0 \
    tensorflow==2.15.0 \
    distrax==0.1.5 \
    flax==0.7.5 \
    ml_collections>=0.1.0 \
    tqdm>=4.60.0 \
    absl-py>=0.12.0 \
    scipy==1.10.0 \
    wandb>=0.12.14 \
    einops>=0.6.1 \
    imageio>=2.31.1 \
    moviepy>=1.0.3 \
    transformers>=4.34.1 \
    tensorflow_hub>=0.14.0 \
    tensorflow_text>=2.13.0 \
    tensorflow_datasets==4.9.2 \
    tensorflow_graphics==2021.12.3 \
    plotly>=5.16.1 \
    matplotlib

# ワークスペースの設定
WORKDIR /workspace
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> ~/.bashrc

# rosdepの初期化
RUN rosdep update
