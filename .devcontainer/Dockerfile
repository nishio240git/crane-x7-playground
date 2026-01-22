FROM osrf/ros:jazzy-desktop-full

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-moveit \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10をインストール（Octo用）
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Octo用のPython 3.10仮想環境を作成
RUN python3.10 -m venv /opt/octo_env
ENV OCTO_VENV=/opt/octo_env

# 仮想環境内にOctoの依存関係をインストール
RUN /opt/octo_env/bin/pip install --upgrade pip && \
    /opt/octo_env/bin/pip install jax==0.4.20 jaxlib==0.4.20 && \
    /opt/octo_env/bin/pip install \
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
RUN echo "# Octoを使う場合は: source /opt/octo_env/bin/activate" >> ~/.bashrc
RUN echo "alias activate-octo='source /opt/octo_env/bin/activate'" >> ~/.bashrc

# rosdepの初期化
RUN rosdep update
