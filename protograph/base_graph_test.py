import pandas as pd
import numpy as np
import torch
Base_graph_filename = 'BaseGraph/wman_N0576_R34_z24.txt'



def cnt_edge_type():
    df=pd.read_csv(Base_graph_filename,header=None,sep=r'\s+')
    np_array=df.values.astype(np.float32)
    print(np_array)

    print(np_array.shape)
    cnt=0
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            if(np_array[i][j]!=-1):
                cnt=cnt+1

    return cnt


print(cnt_edge_type())

print(torch.linspace(-4, 4, 4))

import torch
print(torch.cuda.is_available()) 
# 이게 False라면? -> NVIDIA 드라이버나 CUDA Toolkit 버전이 설치된 PyTorch와 안 맞는 겁니다.