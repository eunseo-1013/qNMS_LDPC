


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





# model 1 bit
b=2

frame = 5000
batch = 100
epoch = 10
test_frame= 10000

iteration_num=20
train_snr=2.0 
learning_rate=0.005




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ",device)



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

'''

def Q(x, eta_sq, qk):
    sum_top=0
    sum_bottom=0
    for i in range(2**b):
        q=qk[i]
        upper=((x-q)**2)/(2*(eta_sq**2))
        sum_top += q*torch.exp(-upper)
        sum_bottom+=torch.exp(-upper)
    #print("최소값 ....", sum_bottom.min())
    #print("최소값 .... upper", sum_top.min())
    return sum_top/(sum_bottom+1e-12)


'''
# 병렬로 바꾸기
def Q(x, eta, qk):
    logits = -((x.unsqueeze(-1).to(device) - qk.to(device))**2) / (2 * (eta**2) + 1e-12)
    # 파이토치의 최적화된 softmax 사용
    weights = torch.nn.functional.softmax(logits, dim=-1)
    return torch.sum(weights.to(device) * qk.to(device), dim=-1)






#=================== edge type weight =================

Base_graph_filename = 'BaseGraph/wman_N0576_R34_z24.txt'
def cnt_edge_type():
    df=pd.read_csv(Base_graph_filename,header=None,sep=r'\s+')
    np_array=df.values.astype(np.float32)
    print(np_array)

    print(np_array.shape)
    position=[]
    cnt=0
    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            if(np_array[i][j]!=-1):
                cnt=cnt+1
                position.append(tuple(i,j))


    return cnt,np_array.shape,position


sharing_weight_cnt,sharing_weight_shape,sharing_weight_position=cnt_edge_type()




def c_to_v(M, alpha,beta):
   
    
    abs_M = torch.abs(M)
    masked_abs_M = torch.where(H == 1, abs_M, torch.tensor(float('inf'), device=device))
    
 
    min_vals, min_indices = torch.topk(masked_abs_M, k=2, dim=2, largest=False)
    min_val1 = min_vals[:, :, 0:1] # (batch, M, 1)
    min_val2 = min_vals[:, :, 1:2] # (batch, M, 1)
    min_idx1 = min_indices[:, :, 0:1] # 가장 작은 값의 위치(열 인덱스)
    
   
    node_indices = torch.arange(N, device=device).reshape(1, 1, N)
    E_abs = torch.where(node_indices == min_idx1, min_val2, min_val1)
    
    
    signs = torch.sign(M)
    
    valid_signs = torch.where(H == 1, signs, torch.ones_like(signs))
    row_sign_prod = torch.prod(valid_signs, dim=2, keepdim=True)
    E_sign = row_sign_prod * valid_signs 
    '''

    E=[]
    for i in range(sharing_weight_shape[0]):
        E_col=[]
        for j in range(sharing_weight_shape[1]):
            alpha_z=torch.ones(Z,Z)*alpha[i][j]
            beta_z=torch.ones(Z,Z)*beta[i][j]
            E_z=alpha_z * E_sign * torch.max(torch.zeros_like(Z,Z),E_abs[i*Z,j*Z]-beta_z)
            E_col.concat(E_z,dim=1)
        E.concat(E_col,dim=0)
        '''
    
    # 위의 코드 병렬화
    alpha_expanded = alpha.repeat_interleave(Z, dim=0).repeat_interleave(Z, dim=1)
    beta_expanded = beta.repeat_interleave(Z, dim=0).repeat_interleave(Z, dim=1)
    E = alpha_expanded * E_sign * torch.relu(E_abs - beta_expanded)
    
    #E = alpha * E_sign * torch.max(torch.zeros_like(E_abs),E_abs-beta)
    return E * H 
# decoder 1

class NMS(nn.Module):
    def __init__(self,it=3):
        super().__init__()
        self.iteration=it
        self.alpha=nn.Parameter(torch.ones(sharing_weight_shape[0],sharing_weight_shape[1],self.iteration)*0.7) # edge-type 별 가중치 적용
        self.beta=nn.Parameter(torch.ones(sharing_weight_shape[0],sharing_weight_shape[1],self.iteration)*0.05)# edge-type 별 가중치 적용
        #self.eta=nn.Parameter(torch.ones(self.iteration)*0.7) # iter 별 가중치 적용
        num_levels = 2**b
        #uniform 초기값
        #qk_init = torch.linspace(-4, 4, num_levels) 
        #self.qk = nn.Parameter(qk_init)
    def forward(self,r): #llr 계산
        M=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  v -> c ( M(n-k)  x N)
        E=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  c -> v
        M=initial_M(M,r)
        for iter in range(self.iteration): # 한 프레임당 반복 수
            # c -> v 
            E=c_to_v(M,alpha=self.alpha[:,iter],beta=self.beta[:,iter])
            #E=Q(E,self.eta[iter],self.qk)
            M=update_M(E, r)
            #M=Q(M,self.eta[iter],self.qk)
        return r + torch.sum(E,dim=1)
    



model=NMS(it=iteration_num)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_fn =  nn.BCEWithLogitsLoss()   # 이거 frame 으로 바꿔야함

torch.manual_seed(42)




SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0, 4.5, 5.0]
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

'''
#--------------------------------------- nms 디코딩--------------------------------

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
        loss=loss_fn(llr_hat[:,:],orignal_code)
        loss.backward()
        optimizer.step() 
   
    print("epoch : " , i, "updated alpha : ", model.alpha.data)  # 1epoch 당  알파 업데이트 값
    print("epoch : " , i, "updated beta : ", model.beta.data)  # 1epoch 당  알파 업데이트 값
    print("epoch : " , i, "updated qk : ", model.qk.data)  # 1epoch 당  알파 업데이트 값
    print("epoch : " , i, "updated eta : ", model.eta.data)  # 1epoch 당  알파 업데이트 값
    

'''

print("updated alpha : ", model.alpha.data)  # 최종  알파 업데이트 값


print("test start!") 








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
            #print(final_llr_hat)
            # hard decision
            Z=hard_decision(final_llr_hat)
            mask=(orignal_code == Z)
            ber = ber+ (orignal_code[:,:K]!=Z[:,:K]).sum().item()
        ber=ber/(K*test_frame)
        BER_array.append(ber)
        print("SNR :",snr,"BER :",ber)

print(BER_array)

