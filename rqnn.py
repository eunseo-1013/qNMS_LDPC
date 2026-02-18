
# Nand flash 의 비선형~~ 

#-----------

from llr_q_iteration_w import RREF
from llr_q_iteration_w import H_to_tensor
from llr_q_iteration_w import make_G_using_H
from llr_q_iteration_w import make_k_bit
from llr_q_iteration_w import AWGN_re_inital_r
from llr_q_iteration_w import initial_M
from llr_q_iteration_w import c_to_v
from llr_q_iteration_w import cal_L
from llr_q_iteration_w import hard_decision
from llr_q_iteration_w import update_M
from llr_q_iteration_w import Q

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
print("device = ",device)


class NMS(nn.Module):
    def __init__(self,it=3):
        super().__init__()
        self.iteration=it
        self.alpha=nn.Parameter(torch.ones(self.iteration)*0.7) # iter 별 가중치 적용
        self.beta=nn.Parameter(torch.ones(self.iteration)*0.05) # iter 별 가중치 적용
        self.eta=nn.Parameter(torch.ones(self.iteration)*0.7) # iter 별 가중치 적용
        num_levels = 2**b
        #uniform 초기값
        qk_init = torch.linspace(-4.0, 4.0, num_levels) 
        self.qk = nn.Parameter(qk_init)
    def forward(self,r): #llr 계산
        M=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  v -> c ( M(n-k)  x N)
        E=torch.zeros(size=(batch,H.shape[0],H.shape[1]),device=device) #  c -> v
        M=initial_M(M,r)
        for iter in range(self.iteration): # 한 프레임당 반복 수
            # c -> v 
            E=c_to_v(M,alpha=self.alpha[iter],beta=self.beta[iter])
            E=Q(E,self.eta[iter],self.qk)
            M=update_M(E, r)
            M=Q(M,self.eta[iter],self.qk)
        return r + torch.sum(E,dim=1)
    



b=4
frame = 100
batch = 10
epoch = 10
test_frame= 10000
iteration_num=5
train_snr=2.0 
learning_rate=0.005


model=NMS(it=iteration_num)
model_2=NMS(it=iteration_num)
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
optimizer_2=torch.optim.Adam(model_2.parameters(),lr=learning_rate)
loss_fn =  nn.BCEWithLogitsLoss(reduction='none')   # 프레임 별로
loss_fn_2 =  nn.BCEWithLogitsLoss(reduction='none') 



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


def sydrome_check(llr_hat,original_code):
    Z=hard_decision(llr_hat)
    errors_per_frame = (original_code != Z).sum(dim=1)
    error_frame = (errors_per_frame > 0)
    return error_frame

#---------------------------------------- nms 디코딩--------------------------------



train_data_2=[]
train_data_2_answer=[]
model.train()
for i in range(epoch): 
    r,orignal_code=0
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
        loss=loss_fn(llr_hat,orignal_code.float())
        loss=loss.mean(dim=1) 
        error_frames=sydrome_check(llr_hat,orignal_code)  
        loss_frame=loss[error_frames].mean() # step 마다 체크해서 틀린 프레임 로스만 반영! <- 맞춤 frame 이 있을까 ㅠㅠ?
        loss_frame.backward()
        optimizer.step()
        if(i == (epoch -1)):
            train_data_2.append(r[error_frames].detach())
            train_data_2_answer.append(orignal_code[error_frames].detach())
    print("epoch : " , i, "updated alpha : ", model.alpha.data)  # 1epoch 당  알파 업데이트 값
    print("epoch : " , i, "updated beta : ", model.beta.data)  # 1epoch 당  알파 업데이트 값
    print("epoch : " , i, "updated qk : ", model.qk.data)  # 1epoch 당  알파 업데이트 값
    print("epoch : " , i, "updated eta : ", model.eta.data)  # 1epoch 당  알파 업데이트 값


print("updated alpha : ", model.alpha.data)  # 최종  알파 업데이트 값


if len(train_data_2) > 0:
    train_data_2 = torch.cat(train_data_2, dim=0)
    train_data_2_answer = torch.cat(train_data_2_answer, dim=0)

print(len(train_data_2))
'''
for i in range(epoch):
    for j in range(len(train_data_2)):
        r=train_data_2[j,:,:]
        # Neural
        optimizer.zero_grad()
        llr_hat= - model_2(r)
        loss=loss_fn(llr_hat,orignal_code.float())
        loss=loss.mean(dim=1) 
        error_frames=sydrome_check(llr_hat,orignal_code)  
        loss_frame=loss[error_frames].mean() # step 마다 체크해서 틀린 프레임 로스만 반영! <- 맞춤 frame 이 있을까 ㅠㅠ?
        loss.backward()
        optimizer.step()


'''



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
            ber = ber+ (mask == False).sum().item()
        ber=ber/(N*test_frame)
        BER_array.append(ber)
        print("SNR :",snr,"BER :",ber)

print(BER_array)

