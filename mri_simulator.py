#streamlit run mri_simulator.py
#pip install streamlit

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

# Disable PyTorch warnings
torch.set_default_dtype(torch.float32)

st.title("MRI Physics Simulator")

# Sidebar sliders
T1_val = st.sidebar.slider("T1 (ms)", 200, 3000, 1000)
T2_val = st.sidebar.slider("T2 (ms)", 20, 500, 100)
T2s_val = st.sidebar.slider("T2* (ms)", 10, 300, 80)
alpha_deg = st.sidebar.slider("Flip Angle α (°)", 1, 90, 30)
TR_val = st.sidebar.slider("TR (ms)", 100, 5000, 500)
TE_val = st.sidebar.slider("TE (ms)", 5, 200, 20)
b_val = st.sidebar.slider("b-value (s/mm²)", 0, 2000, 1000)
D_val = st.sidebar.slider("Diffusion Coefficient D (mm²/s)", 0.1, 3.0, 1.0)
noise_std = st.sidebar.slider("Noise σ", 0.001, 0.2, 0.05)

# Convert values
T1 = torch.tensor(T1_val / 1000)
T2 = torch.tensor(T2_val / 1000)
T2_star = torch.tensor(T2s_val / 1000)
TR = torch.tensor(TR_val / 1000)
TE = torch.tensor(TE_val / 1000)
alpha_rad = torch.tensor(np.deg2rad(alpha_deg), dtype=torch.float32)
D = torch.tensor(D_val / 1000)
b = torch.tensor(b_val, dtype=torch.float32)
sigma = torch.tensor(noise_std)

# Constants
gamma = 42.577e6
B0 = 3.0
M0 = 1.0
omega_0 = gamma * B0
t = torch.linspace(0, 1, 1000)
TI = torch.linspace(0, 2, 1000)

# MRI signal models
Mz_T1 = M0 * (1 - torch.exp(-t / T1))
Mxy_T2 = M0 * torch.exp(-t / T2)
Mxy_T2_star = M0 * torch.exp(-t / T2_star)
FID = M0 * torch.exp(-t / T2_star) * torch.exp(1j * 2 * np.pi * omega_0 * t)
SpinEcho = M0 * torch.exp(-TE / T2)
GradEcho = M0 * torch.exp(-TE / T2_star)
SPGR = M0 * (torch.sin(alpha_rad) * (1 - torch.exp(-TR / T1))) / \
       (1 - torch.cos(alpha_rad) * torch.exp(-TR / T1)) * torch.exp(-TE / T2_star)
Mz_IR = M0 * (1 - 2 * torch.exp(-TI / T1))
S_DWI = M0 * torch.exp(-b * D)
SNR = M0 / sigma

# Plot
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

axs[0, 0].plot(t.numpy(), Mz_T1.numpy())
axs[0, 0].set_title(f"T1 Recovery (T1={T1_val} ms)")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Mz")

axs[0, 1].plot(t.numpy(), Mxy_T2.numpy(), label="T2")
axs[0, 1].plot(t.numpy(), Mxy_T2_star.numpy(), label="T2*")
axs[0, 1].set_title("T2 / T2* Decay")
axs[0, 1].legend()

axs[1, 0].plot(t.numpy(), torch.real(FID).numpy(), label='Real')
axs[1, 0].plot(t.numpy(), torch.imag(FID).numpy(), label='Imag')
axs[1, 0].set_title("FID Signal")
axs[1, 0].legend()

axs[1, 1].bar(['Spin Echo', 'Grad Echo', 'SPGR'], [SpinEcho.item(), GradEcho.item(), SPGR.item()])
axs[1, 1].set_title("Echo Signal Comparison")

axs[2, 0].plot(TI.numpy(), Mz_IR.numpy())
axs[2, 0].set_title("Inversion Recovery")
axs[2, 0].set_xlabel("TI (s)")
axs[2, 0].set_ylabel("Mz")

axs[2, 1].bar(['DWI Signal', 'SNR'], [S_DWI.item(), SNR.item()])
axs[2, 1].set_title("DWI Signal & SNR")

plt.tight_layout()
st.pyplot(fig)
