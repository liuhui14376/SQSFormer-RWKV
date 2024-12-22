import torch
import time
import multiprocessing as mp
import sys



def print_system_info():
    print("系统信息:")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("CUDA 不可用。请检查您的PyTorch安装是否包含CUDA支持。")


def test_gpu(gpu_id, matrix_size=10000):
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)

        matrix1 = torch.randn(matrix_size, matrix_size).to(device)
        matrix2 = torch.randn(matrix_size, matrix_size).to(device)

        torch.mm(matrix1, matrix2)
        torch.cuda.synchronize()

        start_time = time.time()
        result = torch.mm(matrix1, matrix2)
        torch.cuda.synchronize()
        end_time = time.time()

        execution_time = end_time - start_time
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"GPU {gpu_id} ({gpu_name}) 执行时间: {execution_time:.4f} 秒")

        return execution_time
    except Exception as e:
        print(f"测试 GPU {gpu_id} 时出错: {str(e)}")
        return None


def test_cpu(matrix_size=5000):
    matrix1 = torch.randn(matrix_size, matrix_size)
    matrix2 = torch.randn(matrix_size, matrix_size)

    start_time = time.time()
    result = torch.mm(matrix1, matrix2)
    end_time = time.time()

    return end_time - start_time


def run_tests():
    print_system_info()

    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 个GPU")

    if num_gpus == 0:
        print("没有检测到可用的GPU。将只进行CPU测试。")
        cpu_time = test_cpu()
        print(f"CPU 执行时间: {cpu_time:.4f} 秒")
        return

    results = []
    for i in range(num_gpus):
        results.append(test_gpu(i))

    print("\n性能总结:")
    for i, time in enumerate(results):
        if time is not None:
            print(f"GPU {i}: {time:.4f} 秒")


import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建模型实例
model = SimpleNet()

# 检查是否有可用的 GPU，如果有就使用第一个 GPU，否则使用 CPU
device = torch.device("cuda:0")
print("Using device:", device)

# 将模型移动到设备
model = model.to(device)

# 创建一些示例数据（这里使用随机生成的数据）
input_data = torch.randn(5, 10).to(device)
target = torch.randn(5, 1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    output = model(input_data)

    # 计算损失
    loss = criterion(output, target)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印每个epoch的损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

print("Training finished!")


if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'
    mp.set_start_method('spawn')
    run_tests()