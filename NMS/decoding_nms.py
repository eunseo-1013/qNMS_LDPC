from ldpc import make_k_bit
from ldpc import H_to_tensor
from ldpc import RREF
from ldpc import make_G_using_H
import pandas as pd
import numpy as np
import torch





# H 파일 열기 ( 이미 shifting o )
#wman_N0576_R34_z24.txt
#wman_N1152_R12_z48.txt
#wman_N1152_R12_z72.txt


filename="wman_N0576_R34_z24.txt"
N=int(filename[6:10])
K=N*int(filename[12:13])/int(filename[13:14])
K=int(K)

print("N:", N ,", K :" , K)

K_bit = make_k_bit(K)
H=H_to_tensor(filename)
G=make_G_using_H(RREF(H),K)

code = K_bit@G
code=(code%2)
code=code.float()

#print((code@H.T)%2) # 인코딩 굳

SNR=[1.0,2.0,3.0]
snr=50

# AWGN 환경 통과 <- 재미나이 헬프~~ 추후 재코딩 해야함
signal_power=torch.mean(code**2)
snr_linear=10**(snr/10)
noise_power=signal_power/snr_linear
sigma = torch.sqrt(noise_power) # 표준편차

noise=torch.randn_like(code)*sigma
received_signal = code + noise
print(torch.min(received_signal))