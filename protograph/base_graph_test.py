import pandas as pd
import numpy as np


filename = 'BaseGraph/wman_N0576_R34_z24.txt'


df=pd.read_csv(filename,header=None,sep=r'\s+')
np_array=df.values.astype(np.float32)
print(np_array)

print(np_array.shape)
cnt=0
for i in range(np_array.shape[0]):
    for j in range(np_array.shape[1]):
        if(np_array[i][j]!=-1):
            cnt=cnt+1

print(cnt)
print(cnt*24)


import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())