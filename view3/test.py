import sys
import torch

def main():
    # 1. Python 版本
    print("Python 版本:", sys.version)

    # 2. PyTorch 版本
    print("PyTorch 版本:", torch.__version__)

    # 3. CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print("CUDA 可用:", cuda_available)

    # 4. CUDA 版本
    if cuda_available:
        print("CUDA 版本:", torch.version.cuda)
        print("可用 GPU 数量:", torch.cuda.device_count())
        print("当前 GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA 不可用，可能未安装或驱动不匹配")

    # 5. PyTorch 是否能在 GPU 上运行一个简单张量运算
    try:
        x = torch.rand(3, 3).cuda() if cuda_available else torch.rand(3, 3)
        y = x * 2
        print("GPU 张量计算测试成功，结果示例:\n", y)
    except Exception as e:
        print("GPU 测试失败:", e)

if __name__ == "__main__":
    main()
