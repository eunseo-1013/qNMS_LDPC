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
code = 1 - 2*code # bpsk 처리 안했었네..
#print((code@H.T)%2) # 인코딩 확인 굳

SNR=[1.0,2.0,3.0,4.0,50]
snr=SNR[4]

# AWGN 환경 통과  <- 재미나이 헬프~~ 
signal_power=torch.mean(code**2)
snr_linear=10**(snr/10)
noise_power=signal_power/snr_linear
sigma = torch.sqrt(noise_power) # 표준편차

noise=torch.randn_like(code)*sigma
received_signal = code + noise




#nms decoder

r=((2/sigma**2)*received_signal).squeeze(0) # 사전 정보 (1 x n )

M=torch.zeros(size=H.shape) #  v -> c ( M(n-k)  x N)
E=torch.zeros(size=H.shape) #  c -> v


# 초기 v -> c 계산
for c in range(M.shape[0]):
    for v in range(M.shape[1]):
        if(H[c,v]==1):
            M[c,v]=r[v]

print(M.shape)
print(M)

iteration=10

for _ in range(iteration):
    # c -> v 
    for c in range(E.shape[0]):
        find=torch.where(H[c,:]==1)[0]   # 체크노드에 연결된 v 
        if(len(find)==1):
            print(c)
            continue
        for v in find:
            find_e=find[find!=v] # 자기자신 제외
            min_v=torch.min(torch.abs(M[c,find_e]))
            sgn=torch.prod(torch.sign(M[c,find_e]))
            E[c,v]=sgn*min_v

    #print(E)

    L=torch.zeros(N)
    for v in range(E.shape[1]):
        find=torch.where(H[:,v]==1)[0] # 변수노드에 연결된 모든 체크노드
        total=torch.sum(E[find,v])
        L[v]=total+r[v]


    # hard decision
    Z=torch.zeros(N,dtype=int)
    for v in range(N):
        if L[v] < 0:
            Z[v] = 1


    print(Z)
    # 확인
    sydrome=(H@Z)%2

    if(torch.all(sydrome==0)):
        print("신드롬 통과~")
        break
    else:
        print("풉!")
        
    # 통과 안됐을때! M 정보 업데이트
    for v in range(M.shape[1]):
        find=torch.where(H[:,v]==1)[0]
        for c in find:
            other=find[find!=c] # 자기자신 제외
            M[c,v]=torch.sum(E[other,v])+r[v]



