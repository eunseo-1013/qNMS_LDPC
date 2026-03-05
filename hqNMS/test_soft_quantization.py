import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Q_soft(x, eta, qk):
    qk = qk.to(x.device)
    logits = -((x.unsqueeze(-1) - qk)**2) / (2*(eta**2) + 1e-12)
    weights = torch.softmax(logits, dim=-1)
    return torch.sum(weights * qk, dim=-1)

b = 3
eta =0

if b == 3:
    qk = torch.tensor([-4,-3,-2,-1,0,1,2,3])
elif b == 4:
    qk = torch.arange(-8, 8)
elif b == 5:
    qk = torch.arange(-31, 31)



alpha=2
x = torch.linspace(-6, 6, 2000).to(device)
step=2*alpha/4
qk = torch.arange(-alpha, alpha , step)
print(qk)
y_2 = Q_soft(x, eta, qk)
qk = torch.arange(-alpha, alpha ,step/2)
y_3 = Q_soft(x, eta, qk)
print(qk)
qk = torch.arange(-alpha, alpha ,step/4)
print(qk)
y_4 = Q_soft(x, eta, qk)

plt.plot(x.cpu().numpy(), y_2.detach().cpu().numpy(),label="2bit")
plt.plot(x.cpu().numpy(), y_3.detach().cpu().numpy(),label="3bit")
plt.plot(x.cpu().numpy(), y_4.detach().cpu().numpy(),label="4bit")
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("Q(x)")
plt.title("test - Hard Quantization( eta = 0)")
plt.show()