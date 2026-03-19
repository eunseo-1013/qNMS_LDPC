import matplotlib.pyplot as plt
import numpy as np


SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]


#BER_1=[0.02647708333333333, 0.023560185185185184, 0.02076736111111111, 0.017843055555555556, 0.01487662037037037, 0.011365972222222222, 0.0071766203703703705, 0.0024643518518518517, 0.00037476851851851853]
FER_1=[0.2, 0.2, 0.2, 0.2, 0.1993, 0.1943, 0.1599, 0.0718, 0.014]
FER_2=[1.0, 1.0, 1.0, 1.0, 0.9933, 0.9176, 0.5973, 0.1694, 0.0177]

iteration_num=20
train_snr=2.0

plt.figure(figsize=(10, 7))

# semilogy
#plt.semilogy(SNR, BER_1, marker='o', markersize=6, linewidth=1.5,label=" NMS - spatial weight sharing,  init alpha =0.7 init beta = 0.05")
plt.semilogy(SNR, FER_1, marker='o', markersize=6, linewidth=1.5,label=" NMS - spatial weight sharing,  init alpha =0.7 init beta = 0.05")
plt.semilogy(SNR, FER_2, marker='o', markersize=6, linewidth=1.5,label=" NMS - edge - weight sharing,  init alpha =0.7 init beta = 0.05")
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)
plt.title(f'Iteration: {iteration_num}, Train SNR: {train_snr}dB\n SNR - BER ', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='best', fontsize=10)
plt.show()