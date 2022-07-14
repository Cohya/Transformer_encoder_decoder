
import matplotlib.pyplot as plt 
import numpy as np 


# x = np.arange(10)
# n = 1
# d_model = 10
n=10000
d_model=512
position_vec = []

for pos in range(100):
    pos_i = []
    
    for i in range(256):
        even = np.sin(pos/(n**(2*i/d_model)))
        pos_i.append(even)
        odd = np.cos(pos/(n**(2*i/d_model)))
        pos_i.append(odd)
        
    # plt.figure()
    # plt.plot(pos_i)
    
    position_vec.append(pos_i)
    
position_vec = np.asarray(position_vec)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '16'
plt.contourf(position_vec)
plt.colorbar()
plt.ylabel('Index Token')
plt.xlabel('positional encoding')