from ldpc import make_k_bit
from ldpc import H_to_tensor
from ldpc import RREF
from ldpc import make_G_using_H

'''
from msa import AWGN_re_inital_r
from msa import initial_M
from msa import cal_L
from msa import c_to_v
from msa import hard_decision
from msa import update_M
'''

import torch
import torch.nn as nn



def AWGN_re_inital_r(snr,code):
    # AWGN 환경 통과  <- 재미나이 헬프~~ 
    signal_power=torch.mean(code**2)
    snr_linear=10**(snr/10)
    noise_power=signal_power/snr_linear
    sigma = torch.sqrt(noise_power) # 표준편차

    noise=torch.randn_like(code)*sigma
    received_signal = code + noise
    r=((2/sigma**2)*received_signal).squeeze(0) # 사전 정보 (n)
    return r


#nms decoder
def initial_M(M,r):
    # 초기 v -> c 계산
    M=M.clone()
    for c in range(M.shape[0]):
        for v in range(M.shape[1]):
            if(H[c,v]==1):
                M[c,v]=r[v]
    return M


def c_to_v(E,M,alpha=1,beta=0):
    for c in range(E.shape[0]):
        find=torch.where(H[c,:]==1)[0]   # 체크노드에 연결된 v 
        for v in find:
            find_e=find[find!=v] # 자기자신 제외
            min_v=torch.min(torch.abs(M[c,find_e]))
            sgn=torch.prod(torch.sign(M[c,find_e]))
            E[c,v]=sgn*(alpha*min_v-beta)
    return E
def cal_L(L,E):
    for v in range(E.shape[1]):
        find=torch.where(H[:,v]==1)[0] # 변수노드에 연결된 모든 체크노드
        total=torch.sum(E[find,v])
        L[v]=total+r[v]
    return L

def hard_decision(L,Z):
    for v in range(N):
        if L[v] < 0:
            Z[v] = 1
    return Z

def update_M(M,E):
    for v in range(M.shape[1]):
        find=torch.where(H[:,v]==1)[0]
        for c in find:
            other=find[find!=c] # 자기자신 제외
            M[c,v]=torch.sum(E[other,v])+r[v]
    return M




torch.manual_seed(42)

SNR=[1.0,2.0,3.0,4.0,5.0]
iteration=3
filename="wman_N0576_R34_z24.txt"
N=int(filename[6:10])
K=N*int(filename[12])/int(filename[13])
K=int(K)
print("N:", N ,", K :" , K)
frame=50 # 비트 수

#-------------------------------------ldpc 인코딩-------------------------------
H=H_to_tensor(filename)
G=make_G_using_H(RREF(H),K)
#print((code@H.T)%2) # 인코딩 확인 굳
print("------AWGN-----")
print("iteration :",iteration)
BER_array=[]
L=torch.zeros(N)
Z=torch.zeros(N,dtype=int)

class NMS(nn.Module):
    def __init__(self,iteration):
        super().__init__()
        self.iteration=iteration
        self.alpha=nn.Parameter(torch.ones(iteration)*0.7) # iter 별 가중치 적용
    def forward(self,r): #llr 계산
        M=torch.zeros(size=H.shape) #  v -> c ( M(n-k)  x N)
        E=torch.zeros(size=H.shape) #  c -> v
        M=initial_M(M,r)
        for iter in range(self.iteration): # 한 프레임당 반복 수
            # c -> v 
            E=c_to_v(E,M,alpha=self.alpha[iter],beta=0)
            M=update_M(M,E)
        return r + torch.sum(E,dim=0)


model=NMS(iteration=3)
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
loss_fn = nn.BCEWithLogitsLoss()  



#---------------------------------------- nms 디코딩--------------------------------
        

for snr in SNR: # SNR 별
    ber=0
    
    for i in range(frame): # bit_num == 프레임수 
        K_bit = make_k_bit(K)
        code = K_bit@G
        code=(code%2)
        code=code.float()
        orignal_code=code
        code = 1 - 2*code # bpsk 처리 안했었네..
        r=AWGN_re_inital_r(snr,code)
        L=torch.zeros(N)
        Z=torch.zeros(N,dtype=int)

        # Neural
        for epoch in range(10):
            optimizer.zero_grad()
            llr_hat=model(r)
            loss=loss_fn(llr_hat[:K],K_bit.squeeze(0).float())
            loss.backward()
            optimizer.step()
            
        print(model.alpha.data)  
         # hard decision
        hard_decision(llr_hat,Z)
        mask=(orignal_code == Z)
        ber = ber+ (mask == False).sum()

    ber=ber/(N*frame)
    BER_array.append(ber.item())
    print("SNR :",snr,"BER :",ber.item())
    

print(BER_array)


 