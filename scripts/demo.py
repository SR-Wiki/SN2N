import torch


def check_gpu_availability():
    # 检查CUDA是否可用
    is_available = torch.cuda.is_available()
    print(f"CUDA可用: {is_available}")

    if is_available:
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")

        # 显示每个GPU的名称
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # 显示当前使用的GPU
        current_device = torch.cuda.current_device()
        print(f"当前使用的GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")

        # 显示CUDA版本
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("未检测到可用的GPU，将使用CPU进行计算")


if __name__ == "__main__":
    check_gpu_availability()
