import torch


step=1
b=5

def make_qk(b,step):
    range=2**b
    qk=torch.arange(-range/(step*2),(range/(step*2))-step,step)
    return qk

qk_c = make_qk(b,step)  # 여긴 무조건 고정 값! 
print(qk_c)