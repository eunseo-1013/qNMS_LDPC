import matplotlib.pyplot as plt
import numpy as np
SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]


BER_1 = [0.06533064236111111, 0.05818472222222222, 0.051140625, 0.04398559027777778, 0.03657291666666666, 0.027885503472222222, 0.016107725694444443, 0.004667708333333334, 0.00054609375]
BER_2 = [0.1370048611111111, 0.12247731481481482, 0.10808541666666667, 0.09267013888888889, 0.07548912037037037, 0.05284490740740741, 0.025032870370370372, 0.005452314814814815, 0.0004787037037037037]
BER_3 = [0.13076180555555555, 0.11622152777777778, 0.10221712962962963, 0.08754143518518519, 0.07232546296296297, 0.054521759259259256, 0.029656944444444444, 0.007931481481481482, 0.0008414351851851852]
BER_4=[0.13052430555555555, 0.11621643518518518, 0.10215092592592592, 0.08746828703703703, 0.07263935185185186, 0.05600439814814815, 0.03581018518518519, 0.0176125, 0.008288657407407408]
BER_5=[0.1314789351851852, 0.11722060185185185, 0.10313657407407407, 0.08866203703703704, 0.07488541666666666, 0.06120046296296296, 0.047931944444444444, 0.03591435185185185, 0.024226157407407408]
BER_6=[0.13057673611111112, 0.11659166666666666, 0.102284375, 0.08790972222222222, 0.0732390625, 0.05720243055555556, 0.04025190972222222, 0.02317326388888889, 0.010581944444444444]
BER_7=[0.2102138888888889, 0.19140023148148147, 0.17277662037037037, 0.15312430555555556, 0.13245833333333334, 0.10942638888888889, 0.07703055555555556, 0.028843287037037035, 0.003118287037037037]

BER_8=[0.1306085648148148, 0.11610439814814814, 0.10211273148148148, 0.0874412037037037, 0.07246319444444445, 0.05374444444444444, 0.028315740740740742, 0.007131712962962963, 0.0006745370370370371]
#BER_9=
BER_10=[0.13026597222222222, 0.11644074074074075, 0.1018400462962963, 0.08746527777777778, 0.0725699074074074, 0.05436504629629629, 0.029496296296296295, 0.008446296296296295, 0.0008282407407407407]
frame = 5000
batch = 50
epoch = 10
test_frame= 10000

iteration_num=25

train_snr=2.0 



# 로그 그래프

plt.figure(figsize=(10, 7))

# semilogy
plt.semilogy(SNR, BER_1, marker='o', markersize=6, linewidth=1.5,label=" NMS ,  init alpha =0.7 init beta = 0.05")
#plt.semilogy(SNR, BER_2, marker='o', markersize=6, linewidth=1.5,label=" SMS , fixed alpha =0.7 fixed beta = 0.05")
#plt.semilogy(SNR, BER_3, marker='o', markersize=6, linewidth=1.5,label=" MS(float) ")
#plt.semilogy(SNR, BER_7, marker='o', markersize=6, linewidth=1.5,label=" 2bit qMS , fixed eta=0.7 fixed qk= -4~4")
#plt.semilogy(SNR, BER_6, marker='o', markersize=6, linewidth=1.5,label=" 2bit qNMS , init eta=0.7 init qk= -4~4")
#plt.semilogy(SNR, BER_4, marker='o', markersize=6, linewidth=1.5,label=" 3bit qNMS , init eta=0.7 init qk= -4~4")
#plt.semilogy(SNR, BER_5, marker='o', markersize=6, linewidth=1.5,label=" 3bit qNMS , init eta=0.7 init qk= -8~8")
plt.semilogy(SNR, BER_8, marker='o', markersize=6, linewidth=1.5,label=" NMS sharing weight , init alpha =0.7 init beta = 0.05")
#plt.semilogy(SNR, BER_9, marker='o', markersize=6, linewidth=1.5,label="2bit fixed qNMS sharing weight ")
plt.semilogy(SNR, BER_10, marker='o', markersize=6, linewidth=1.5,label=" NMS sharing weight , epoch = 50 init alpha =0.7 init beta = 0.05")
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)
plt.title(f'Iteration: {iteration_num}, Train SNR: {train_snr}dB\n SNR - BER ', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='best', fontsize=10)
plt.show()


# 일반그래프
'''




step=30
plt.figure()
plt.plot(x[::step], q_0[::step], label=f'shaping_factor=0.5')
plt.plot(x[::step], q_1[::step], label=f'shaping_factor=0.05')
plt.plot(x[::step], q_2[::step], label='shaping_factor=0.005')
plt.xlabel("x")
plt.ylabel("Qn(x)")
plt.legend()
plt.title(f'qk = [-1,-1/3,1/3,1] b=2')
plt.grid(True)
plt.show()

'''