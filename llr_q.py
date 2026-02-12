

import torch
import torch.nn as nn
import pandas as pd
import numpy as np




SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]
filename="wman_N0576_R34_z24.txt"
N=int(filename[6:10])
K=N*int(filename[12])/int(filename[13])
K=int(K)
print("N:", N ,", K :" , K)


frame = 5000
batch = 50
epoch = 10
test_frame= 10000

iteration_num=20

learning_rate = 0.005


step = frame // batch # 1epoch 당 몇번 업데이트 ?
test_step = test_frame // batch # 1epoch 당 몇번 업데이트 ?






def H_to_tensor(filename):
    df=pd.read_csv(filename,header=None,sep=r'\s+')
    np_array=df.values.astype(np.float32)
    tr=torch.tensor(np_array)
    print("H shape :", tr.shape)
    return tr

def RREF(H): # H[ A | I ]
    pivot_row=0
    for col in range(H.shape[1]-H.shape[0],H.shape[1]):
    
        #상대적 위치를 찾는다. <- tuple ( tensor , )
        find_1=torch.where((H[pivot_row: , col] ==1))[0] # = tensor ( 1 이 있는 인덱스)
        if(len(find_1)==0): # 1인 인덱스가 x == 이 열이 전부 0 일경우... 패쓰
            continue
        max_idx=find_1[0].item()+pivot_row # 텐서 일반 숫자로 변경 필요
       
        H[[pivot_row,max_idx]]=H[[max_idx,pivot_row]] #swap
        
        # H[pivot_row]= H[pivot_row]/H[pivot_row,col]  <- xor 연산이기에 0,1 만 존재 == 1로만들필요 x
        for i in range(H.shape[0]):
            if(i!=pivot_row and H[i,col] == 1):
                 H[i]=H[i].to(torch.uint8)^H[pivot_row].to(torch.uint8)

        pivot_row+=1
    return H




def make_G_using_H(RREF_H,K):
    I=torch.eye(K,dtype=RREF_H.dtype,device=RREF_H.device)
    A=RREF_H[:,:K]
    print(A.shape)
    G=torch.cat([I,A.t()],dim=1)
    print("G shape:",G.shape)
    return G







device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
print("device = ",device)



def make_k_bit(K,frame): # 돌려볼 비트 만들기  batch  x  k 길이
   orignal_bit=torch.randint(low=0,high=2,size=(frame,K),device=device)
   return (orignal_bit)



def AWGN_re_inital_r(snr,code):
    # AWGN 환경 통과  <- 재미나이 헬프~~ 
    signal_power=torch.mean(code**2)
    snr_linear=10**(snr/10)
    noise_power=signal_power/snr_linear
    sigma = torch.sqrt(noise_power) # 표준편차

    noise=torch.randn_like(code)*sigma
    received_signal = code + noise
    r=((2/sigma**2)*received_signal) # 사전 정보 (n)
    return r






#-------------------------------------ldpc 인코딩-------------------------------
H=H_to_tensor(filename).to(device)
G=make_G_using_H(RREF(H.clone()),K).to(device)
#print((code@H.T)%2) # 인코딩 확인 


#nms decoder
def first_layer(E,r):
    # 초기 엣지값 계산
    E=H.unsqueeze(0)*r.unsqueeze(1)
    return E.to(device)


def Hdi_odd(E, r,alpha_v,beta_v): 

    sum_E = torch.sum(E, dim=1, keepdim=True) # batch x M x N -> batch x 1 x N

    E = r.unsqueeze(1) + sum_E -E  #  R = batch X 1 x N + batch x 1 x N + batch x M x N
    E = E * H.unsqueeze(0) #batch x M x N * (1 x M X N) ==> batch x M x N ( 연결 점에만 값 0)
    E=torch.relu(alpha_v*E+beta_v) + r.unsqueeze(1)

    # + 퀀타 씌워야 완성
    
    return Q(E)


def Hdi_even(M, alpha_c,beta_c):
    # M: (batch, M, N) -> Variable Node에서 Check Node로 온 메시지들
    
    # 1. 절대값과 부호 분리
    abs_M = torch.abs(M).to(device)
    # H=0인 곳은 최솟값 연산에 방해되지 않게 무한대로 채움
    masked_abs_M = torch.where(H == 1, abs_M, torch.tensor(float('inf'), device=device))
    
    # 2. 각 행(Check Node)에서 가장 작은 값 2개를 찾음
    # min_val1: 가장 작은 값, min_val2: 두 번째로 작은 값
    min_vals, min_indices = torch.topk(masked_abs_M, k=2, dim=2, largest=False)
    min_val1 = min_vals[:, :, 0:1] # (batch, M, 1)
    min_val2 = min_vals[:, :, 1:2] # (batch, M, 1)
    min_idx1 = min_indices[:, :, 0:1] # 가장 작은 값의 위치(열 인덱스)
    
    # 3. "나를 제외한 최솟값" 결정
    # 내 위치가 가장 작은 값의 위치와 같다면 두 번째 최솟값을 선택, 아니면 첫 번째 선택
    # masked_abs_M과 같은 크기의 텐서를 만들기 위해 index 비교
    node_indices = torch.arange(N, device=device).reshape(1, 1, N)
    E_abs = torch.where(node_indices == min_idx1, min_val2, min_val1)
    
    
    signs = torch.sign(M)
    # 0이 있으면 곱이 0이 되므로 1로 대체 (H=0인 곳도 1로 대체)
    valid_signs = torch.where(H == 1, signs, torch.ones_like(signs))
    row_sign_prod = torch.prod(valid_signs, dim=2, keepdim=True) # 전체 행의 부호 곱
    E_sign = row_sign_prod * valid_signs # (전체 곱) * (내 부호) = (나를 제외한 곱)
    
    # 5. 최종 메시지 (alpha 적용)
    E =  E_sign.to(device) * torch.relu(alpha_c.to(device)*E_abs+beta_c).to(device)
   
    E=E * H.unsqueeze(0)
   
    E=Q((E))
   
    # 퀀타 추가해야함
    return E.to(device) # H=0인 곳은 다시 0으로 마스킹


def final_output(L,E):
    total=E*H.unsqueeze(0) # 최종 아웃풋
    L=total.sum(dim=1) + r.unsqueeze(0)
    return L




def hard_decision(L):
    Z = (L<0).int()
    return Z

#shaping factor
_eta=0.5
# quantization level
_b=2
'''
def Q(x,eta=_eta,b=_b): # 입력 크기 == batch x M x N or batch x M
    q=torch.arange(1,2**b,device=device,dtype=x.dtype)  #((2^^b)-1 ,) 
    diff=x.unsqueeze(-1)-q # batch  x N x (2^^b)-1.. 이게 이렇게 계산이 돼?
    exp_=torch.exp(-(diff**2)/(2*(eta**2)))
   
    upper= torch.sum(q * exp_,dim=-1)
    lower=torch.sum(exp_,dim=-1)

    return upper/lower  # batch x M x N
 
   
       '''

def Q(x, eta=_eta, b=_b):
    # LLR 범위를 고려하여 -max_val ~ +max_val 구간으로 설정
    max_val = 7.0 # 예시: 4비트라면 -7 ~ 7
    levels = torch.linspace(-max_val, max_val, 2**b, device=device, dtype=x.dtype)
    
    # x: (batch, M, N, 1), levels: (1, 1, 1, num_levels)
    diff = x.unsqueeze(-1) - levels 
    exp_ = torch.exp(-(diff**2) / (2 * (eta**2)))
    
    # Softmax-like weighting
    prob = exp_ / (torch.sum(exp_, dim=-1, keepdim=True) + 1e-9)
    return torch.sum(levels * prob, dim=-1)
def soft_decision(x):
    nsgn=Q(x,eta=_eta,b=2)
    nsgn=2*(nsgn - 1) -1 # bpsk 처리... 모르겠는데??
    return nsgn



class qNMS(nn.Module):
    def __init__(self,it=3):
        super().__init__()
        self.iteration=it
        size=(H.shape[0],H.shape[1],self.iteration)
        self.alpha_c=nn.Parameter(torch.ones(size)) # iter 별 가중치 적용
        self.beta_c=nn.Parameter(torch.zeros(size)) # iter 별 가중치 적용
        self.alpha_v=nn.Parameter(torch.ones(size)) # iter 별 가중치 적용
        self.beta_v=nn.Parameter(torch.zeros(size)) # iter 별 가중치 적용
    def forward(self,r): #llr 계산
        E=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  c -> v
        E=first_layer(E,r)
        for iter in range(self.iteration): # 한 프레임당 반복 수
            E=Hdi_even(E,alpha_c=self.alpha_c[:,:,iter],beta_c=self.beta_c[:,:,iter])
            E=Hdi_odd(E, r,alpha_v=self.alpha_v[:,:,iter],beta_v=self.beta_v[:,:,iter])
        return r + torch.sum(E,dim=1)
    



model=qNMS(it=iteration_num).to(device)


optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_fn =  nn.BCEWithLogitsLoss()





torch.manual_seed(42)







print("------AWGN-----")
print("iteration :",iteration_num)
BER_array=[]
L=torch.zeros(frame,N).to(device)
Z=(torch.zeros(frame,N,dtype=int)).to(device)     
llr_hat=(torch.zeros(frame,N))





#---------------------------------------- nms 디코딩--------------------------------


model.train()
for i in range(epoch): 
    for _ in range(step):
        train_snr = torch.empty(1).uniform_(1.0, 5.0).item() 
        K_bit = make_k_bit(K,batch) # b x k
        code = K_bit.float()@G.float()# (b x k) x (k x n) == (b x n)
        code=(code%2)
        code=code.float()
        orignal_code=code
        code = 1 - 2*code # bpsk 처리 안했었네..
        r=AWGN_re_inital_r(train_snr,code) # b x n
        # Neural
        optimizer.zero_grad()
        llr_hat=  model(r)
        loss=loss_fn(llr_hat[:,:K],K_bit.float())
        loss.backward()
        optimizer.step() 
    #print("epoch : " , i, "updated alpha : ", model.alpha_c.data)  # 1epoch 당  알파 업데이트 값
print("updated alpha : ", model.alpha_c.data)  # 최종  알파 업데이트 값
print("training start!") 
model.eval()



#----------- 성능 평가 -------------
with torch.no_grad(): # 자동 미분 중지
    for snr in SNR:
        ber=0
        for _ in range(test_step):
            K_bit = make_k_bit(K,batch) # f x k
            code = K_bit.float()@G.float() # (f x k) x (k x n) == (f x n)
            code=(code%2).float()
            orignal_code=code
            code = 1 - 2*code # bpsk 처리 안했었네..
            r=AWGN_re_inital_r(snr,code) # f x n
            final_llr_hat = model(r)
            # hard decision
            Z=hard_decision(final_llr_hat)
            mask=(orignal_code == Z)
            ber = ber+ (mask == False).sum().item()
        ber=ber/(N*test_frame)
        BER_array.append(ber)
        print("SNR :",snr,"BER :",ber)

print(BER_array)


 