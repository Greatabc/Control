import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import control as ct


import warnings
warnings.filterwarnings('ignore')
import re
st.header('Second Order')
st.latex(r'''Y_s=\frac{Kp(a) s+ Kp(b)}{\tau^2\:s^2\:+ 2\: \tau\: \zeta\: s +1}''')
numbers = st.text_input("Numerator", [1, 2])
rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", numbers)
num_rr = [eval(i) for i in rr]
numbers = st.text_input("Denumerator", [1, 1, 1])
rr1 = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", numbers)
den_rr = [eval(i) for i in rr1]
tau = np.sqrt(den_rr[0] / den_rr[2])
zeta = den_rr[1] / (2 * tau)
G1s = ct.tf(num_rr, den_rr)
(t, y) = ct.step_response(G1s)
fig, ax = plt.subplots()
ax.plot(t,y,color='blue')
ax.grid()
ax.axvline(color='black')
ax.axhline(color='black')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
st.pyplot(fig)
