import torch


step=2
b=5

def make_threshold(b,step):
    range=(2**b)*step
    qk=torch.arange(-range/(2),(range/(2)),step)
    return qk


qk_threshold=make_threshold(b,step)
print(qk_threshold)

def make_qk(b,step):
    range=(2**b)*step
    qk=torch.arange(-range/(2)+(step/2),(range/(2)),step)
    return qk

# 대표값 

qk_c = make_qk(b,step)  # 여긴 무조건 고정 값! 
print(qk_c)