'''
snr_axis = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
llr_min  = [-11.15, -12.64, -12.56, -13.80, -14.06, -16.29, -184.88, -579.54, -686.47]
llr_max  = [11.32, 12.50, 12.51, 15.02, 14.80, 16.26, 178.20, 564.08, 686.51]
llr_mean = [-0.016, -0.011, -0.052, -0.006, 0.031, 0.046, 0.120, 0.477, -0.261]
llr_std  = [3.49, 3.73, 4.14, 4.69, 5.24, 5.98, 32.19, 175.07, 302.33]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

# 1. Min-Max 범위를 영역으로 표시 (Fill between)
ax1.fill_between(snr_axis, llr_min, llr_max, color='gray', alpha=0.2, label='LLR Range (Min-Max)')
ax1.plot(snr_axis, llr_mean, marker='d', color='black', label='LLR Mean')
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('LLR Value Range', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', alpha=0.6)

# 2. Std(표준편차)를 보조축에 표시 (값이 너무 튀어서 따로 보는 게 좋음)
ax2 = ax1.twinx()
ax2.plot(snr_axis, llr_std, marker='o', color='red', linewidth=2, label='LLR Std (Volatility)')
ax2.set_ylabel('Standard Deviation (Std)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('LLR Statistics Variation by SNR', fontsize=14)
fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()'''

import matplotlib.pyplot as plt
import numpy as np

# 데이터 설정
qk_history = np.array([
    [-3.9529, -2.1271,  0.1252,  1.8765],
    [-4.0336, -2.0138,  0.0125,  1.8808],
    # ... 중간 생략 (보내주신 데이터 전체 사용) ...
    [-4.4487, -1.7885, -0.0102,  2.1990],
    [-4.0268, -2.5404,  0.0010,  2.4172],
    [-4.2118, -2.1072,  0.0003,  2.1149]
])

iterations = np.arange(1, len(qk_history) + 1)

plt.figure(figsize=(10, 7))

# 1. Level 1 (Min)과 Level 4 (Max)만 그리기
plt.plot(iterations, qk_history[:, 0], marker='s', color='#EF4444', 
         label='Level 1 (Min)', linewidth=2, markersize=5)
plt.plot(iterations, qk_history[:, 3], marker='s', color='#3B82F6', 
         label='Level 4 (Max)', linewidth=2, markersize=5)

# 2. 핵심: Y축에 대칭 로그(symlog) 적용
# linthresh는 어느 범위까지 '선형(Linear)'으로 보여줄지 결정합니다. 
# 0 근처의 아주 작은 변화를 무시하지 않기 위해 0.1 정도로 설정합니다.
plt.yscale('symlog', linthresh=1.0)

# 3. 디자인 디테일
plt.axhline(0, color='black', linewidth=1.5, alpha=0.8)
plt.title('Evolution of $q_k$ Range (Symmetric Log Scale)', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Iteration Number', fontsize=12)
plt.ylabel('Quantization Level (Log Space)', fontsize=12)

# Y축 그리드를 로그 스케일에 맞게 촘촘히 표시
plt.grid(True, which="both", ls="--", alpha=0.4)

# Y축 레이블 가독성 (로그 스케일이라 수치가 1, 10, 100 단위로 보임)
plt.yticks([-10, -5, -2, -1, 0, 1, 2, 5, 10])

plt.legend(loc='best', frameon=True)
plt.tight_layout()
plt.show()