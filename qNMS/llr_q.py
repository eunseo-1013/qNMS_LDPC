import torch

if torch.cuda.is_available():
    print("✅ GPU 사용 가능")
 
else:
    print("❌ GPU 사용 불가 (CPU 사용 중)")
