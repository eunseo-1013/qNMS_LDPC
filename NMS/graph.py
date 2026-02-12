import matplotlib.pyplot as plt
import numpy as np
SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]


BER_1 =[0.26168576388888887, 0.23311215277777778, 0.20492708333333334, 0.17676944444444445, 0.14800763888888888, 0.11557256944444444, 0.07292604166666666, 0.030048263888888888, 0.009862847222222223]
BER_2 = [0.26132256944444443, 0.23273888888888888, 0.2045625, 0.1759423611111111, 0.14629166666666665, 0.11154201388888889, 0.06443090277777777, 0.018670833333333334, 0.002184375]
BER_3 =[0.2613447916666667, 0.23275451388888888, 0.20456909722222222, 0.175965625, 0.14628090277777778, 0.11114340277777777, 0.06370347222222222, 0.019060763888888888, 0.0027631944444444444]

BER_q1=[0.13108159722222223, 0.11748958333333333, 0.10442013888888889, 0.09139409722222222, 0.07886631944444444, 0.06733854166666667, 0.05669791666666667, 0.046489583333333334, 0.03790798611111111]

frame = 5000
batch = 50
epoch = 10
test_frame= 10000

iteration_num=20

train_snr=2.0 



# 로그 그래프

plt.figure(figsize=(10, 7))

# semilogy를 사용하면 Y축이 자동으로 로그 스케일이 됩니다.
plt.semilogy(SNR, BER_q1, marker='o', markersize=6, linewidth=1.5)


plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)
plt.title(f'Iteration: {iteration_num}, Train SNR: {train_snr}dB\n SNR - BER (Log Scale)', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='best', fontsize=10)
plt.show()


# 일반그래프
'''
plt.figure()
plt.plot(SNR, BER_1, marker='o', label=f'alpha_init = 0.7 ,beta_init = 0')
plt.plot(SNR, BER_2, marker='s', label=f'alpha_init = 0.7 ,beta_init = 0.2')
plt.plot(SNR, BER_3, marker='^', label='alpha_init = 0.7 ,beta_init = 0.05')
plt.xlabel("SNR (db)")
plt.ylabel("BER")
plt.legend()
plt.title(f'iter = {iteration_num} ,train_snr = {train_snr} \n alpha,beta factor - iteration NMS')
plt.grid(True)
plt.show()
'''
