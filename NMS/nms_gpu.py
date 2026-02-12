#from ldpc import make_k_bit
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


#nms decoder
def initial_M(M,r):
    # 초기 v -> c 계산
    M=H.unsqueeze(0)*r.unsqueeze(1)
    return M.to(device)


'''
def c_to_v(E,M,alpha=1,beta=0):
    mask=(H==1)
    eye=1-torch.eye(N-K,dtype=torch.bool,device=device)
    concat_eye= torch.ones((N-K, K), dtype=torch.bool,device=device)
    eye=torch.cat([eye,concat_eye],dim=1)
    mask=mask&eye
    

    except_self_M = mask.unsqueeze(0) * M
    min_val,min_idx = torch.min(torch.abs(except_self_M),dim=1)
    sgn = torch.prod(torch.where(mask,M,torch.ones_like(M)),dim=1)  # condition 이랑 같은 크기로 맞춰야함
    E = alpha* sgn * ( min_val -beta )
    return E

'''

def c_to_v(M, alpha=1,beta=0):
    # M: (batch, M, N) -> Variable Node에서 Check Node로 온 메시지들
    
    # 1. 절대값과 부호 분리
    abs_M = torch.abs(M)
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
    
    # 4. 부호 계산 (나를 제외한 나머지 모든 원소의 부호 곱)
    signs = torch.sign(M)
    # 0이 있으면 곱이 0이 되므로 1로 대체 (H=0인 곳도 1로 대체)
    valid_signs = torch.where(H == 1, signs, torch.ones_like(signs))
    row_sign_prod = torch.prod(valid_signs, dim=2, keepdim=True) # 전체 행의 부호 곱
    E_sign = row_sign_prod * valid_signs # (전체 곱) * (내 부호) = (나를 제외한 곱)
    
    # 5. 최종 메시지 (alpha 적용)
    E = alpha * E_sign * torch.max(torch.zeros_like(E_abs),E_abs-beta)
    return E * H # H=0인 곳은 다시 0으로 마스킹


def cal_L(L,E):
    total=E*H.unsqueeze(0)
    L=total.sum(dim=1) + r.unsqueeze(0)
    return L

def hard_decision(L):
    Z = (L<0).int()
    return Z

'''

def update_M(M,E):
    mask=(H==1)
    eye=1-torch.eye(N-K,dtype=torch.bool,device=device)
    concat_eye= torch.ones((N-K, K), dtype=torch.bool,device=device)
    eye=torch.cat([eye,concat_eye],dim=1)
    mask=mask&eye
    except_self_E = mask.unsqueeze(0) * E

    M=M.sum(except_self_E,dim=1)+r
    return M
    '''
def update_M(E, r):

    sum_E = torch.sum(E, dim=1, keepdim=True)
    M = r.unsqueeze(1) + sum_E - E
    M = M * H.unsqueeze(0) 
    
    return M






class NMS(nn.Module):
    def __init__(self,it=3):
        super().__init__()
        self.iteration=it
        self.alpha=nn.Parameter(torch.ones(self.iteration)*0.7) # iter 별 가중치 적용
        self.beta=nn.Parameter(torch.ones(self.iteration)*0.05) # iter 별 가중치 적용
    def forward(self,r): #llr 계산
        M=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  v -> c ( M(n-k)  x N)
        E=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  c -> v
        M=initial_M(M,r)
        for iter in range(self.iteration): # 한 프레임당 반복 수
            # c -> v 
            E=c_to_v(M,alpha=self.alpha[iter],beta=self.beta[iter])
            M=update_M(E, r)
        return r + torch.sum(E,dim=1)
    



frame = 5000
batch = 50
epoch = 1
test_frame= 10000

iteration_num=20

train_snr=2.0 
learning_rate=0.005

model=NMS(it=iteration_num)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_fn =  nn.BCEWithLogitsLoss()  



torch.manual_seed(42)



SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]
filename="wman_N0576_R34_z24.txt"
N=int(filename[6:10])
K=N*int(filename[12])/int(filename[13])
K=int(K)
print("N:", N ,", K :" , K)


step = frame // batch # 1epoch 당 몇번 업데이트 ?
test_step = test_frame // batch # 1epoch 당 몇번 업데이트 ?


#-------------------------------------ldpc 인코딩-------------------------------
H=H_to_tensor(filename).to(device)
G=make_G_using_H(RREF(H.clone()),K).to(device)
#print((code@H.T)%2) # 인코딩 확인 




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
        K_bit = make_k_bit(K,batch) # f x k
        code = K_bit.float()@G.float()# (f x k) x (k x n) == (f x n)
        code=(code%2)
        code=code.float()
        orignal_code=code
        code = 1 - 2*code # bpsk 처리 안했었네..
        r=AWGN_re_inital_r(train_snr,code) # f x n
        # Neural
        optimizer.zero_grad()
        llr_hat= - model(r)
        loss=loss_fn(llr_hat[:,:K],K_bit.float())
        loss.backward()
        optimizer.step() 
    print("epoch : " , i, "updated alpha : ", model.alpha.data)  # 1epoch 당  알파 업데이트 값
print("updated alpha : ", model.alpha.data)  # 최종  알파 업데이트 값
print("training start!") 
model.eval()



#----------- 성능 평가 -------------
with torch.no_grad(): # 자동 미분 중지.. 속도 빠르게 할려고
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


 