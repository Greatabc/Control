import streamlit as st
import control as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d



import warnings
warnings.filterwarnings('ignore')
import re
st.header('Second Order')
st.latex(r'''Y_s=\frac{Kp(a) s+ Kp(b)}{\tau^2\:s^2\:+ 2\: \tau\: \zeta\: s +1}''')
c7, c8 = st.columns((1, 4))
with c7:
    numbers = st.text_input("Numerator", [1, 2])
    rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", numbers)
    num_rr = [eval(i) for i in rr]
    numbers = st.text_input("Denumerator", [1, 1, 1])
    rr1 = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", numbers)
    den_rr = [eval(i) for i in rr1]
    tau = np.sqrt(den_rr[0] / den_rr[2])
    zeta = den_rr[1] / (2 * tau)

    def Calculate_second(num_rr, den_rr, tau, zeta):
        G1s = ct.tf(num_rr, den_rr)
        (t, y) = ct.step_response(G1s)
        OS = np.exp(-(np.pi * zeta) / (np.sqrt(1 - zeta ** 2)))
        DR = (OS) ** 2
        W = np.sqrt((1 - zeta ** 2) / tau)
        P = (2 * np.pi * tau) / (np.sqrt((1 - zeta ** 2)))
        Phi = np.arctan(np.sqrt((1 - zeta ** 2)) / zeta ** 2)
        ST = 4 / (zeta * W)
        if len(num_rr) == 2:
            UV = num_rr[1] / den_rr[2]
            coeff = np.polyfit(t, y, 11)
            xn = np.linspace(0, t[-1], 100)
            yn = np.poly1d(coeff)
            yr = yn(xn)
            max_value = np.argmax(yn(xn))
            interp_func = interp1d(yr[0:max_value], xn[0:max_value])
            TR = interp_func(UV)
        else:
            UV = num_rr[0] / den_rr[2]
            TR = float(
                fsolve(lambda x: (1 / np.sqrt((1 - zeta ** 2))) * np.exp(-zeta * x / tau) * np.sin(W * x + Phi), 0.001))
        MV = OS + UV
        st.write(f'OverShoot = {OS: .3f}')
        st.write(f'Ultimate Value = {UV: .3f}')
        st.write(f'Decay Ratio = {DR: .3f}')
        st.write(f'Max Value = {MV: .3f}')
        st.write(f'Periode = {P: .3f}')
        st.write(f'W = {W: .3f}')
        st.write(f'Phi = {Phi: .3f}')
        st.write(f'Rise time = {TR: .3f}')
        st.write(f'Settling Time = {ST: .3f}')
    Calculate_second(num_rr, den_rr, tau, zeta)
with c8:

    G1s=ct.tf(num_rr,den_rr)
    (t,y)=ct.step_response(G1s)
    fig, ax = plt.subplots()
    ax.plot(t,y,color='blue')
    ax.grid()
    ax.axvline(color='black')
    ax.axhline(color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)
    tau=np.sqrt(den_rr[0]/den_rr[2])
    zeta=den_rr[1]/(2*tau)


st.subheader('Definitions')
c3, c4 = st.columns((2, 4))
with c3:
    st.write('**Rise Time: (RT)**  Time needed to reach first time ultimate value')

with c4:
    st.latex(r'''\phi=tan^{-1}\big | \frac{\sqrt{(1-\zeta^2)}}{\zeta^2} \big | ''')
    st.latex(r'''\omega= \frac{\sqrt{(1-\zeta^2)}}{\tau}''')
    st.latex(r'''0=\frac{1}{\sqrt{(1-\zeta^2)}}e^{\frac{-\zeta\ t_R}{\tau}}\ sin(\omega \ t_R + \phi)''')

c5, c6 = st.columns((2, 4))
with c5:
    st.write('**Peak Time:** Time required for the output to reach its first maximum value')
    st.markdown('**Overshoot: (OS) =** Height of peak devided UV')
    st.markdown('**Decay Ratio: (DR) =** Second peak devided by first peak ')
    st.markdown('**Period of Oscillation:** Time between two successive peaks.')
with c6:
    st.latex(r'''t_p=\frac{\pi\ \tau }{\sqrt{(1-\zeta^2)}}''')
    st.latex(r'''OS=e^{\frac{-\pi\ \zeta}{\sqrt{(1-\zeta^2)}}}''')
    st.latex(r'''DR=OS^2''')
    st.latex(r'''P=\frac{2\ \pi\ \tau }{\sqrt{(1-\zeta^2)}}''')

st.subheader('Effect of Variables')
st.latex(r'''Y_s=\frac{Kp(a) s+ Kp(b)}{\tau^2\:s^2\:+ 2\: \tau\: \zeta\: s +1}''')
c1, c2 = st.columns((1, 4))
with c1:
    tau = st.slider('Enter tau value', 0.0, 30.0,1.0)
    zeta = st.slider('Enter zeta value', 0.0, 2.0,0.1)
    Kp_a = st.slider('Enter Kp(a) value', 0.0, 30.0,0.0)
    Kp_b = st.slider('Enter Kp(b) value', 0.0, 30.0,1.0)
with c2:
    num_rd=[Kp_a,Kp_b]
    den_rd=[tau**2,2*tau*zeta,1]
    G2s=ct.tf(num_rd,den_rd)
    (t,y)=ct.step_response(G2s)
    fig, ax = plt.subplots()
    ax.plot(t,y,color='blue')
    ax.grid()
    ax.axvline(color='black')
    ax.axhline(color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

