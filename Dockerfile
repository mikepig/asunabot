# 使用官方 PyTorch 镜像作为基础
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 创建工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y git curl

# 拷贝文件
COPY requirements.txt .
COPY app.py .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建非 root 用户防止写入 / 权限错误
RUN useradd -m user
USER user
ENV HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH \
    HOME=/home/user

# 切换目录并暴露端口
WORKDIR /home/user/app
COPY --chown=user . .

EXPOSE 7860

# 启动程序
CMD ["python", "app.py"]
