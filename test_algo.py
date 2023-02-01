import numpy as np
import matplotlib.pyplot as plt
import math

phi = (np.sqrt(5) - 1)/2
ll = 80

phi_list= [1, phi]
for n in range(2,ll):
    phi_list.append(phi_list[n-2]-phi_list[n-1])

print(phi_list)

phi_n = []
for x in range(ll):
    phi_n.append(phi**x)

print('compare', phi_n)

print(np.array(phi_n) - np.array(phi_list))



fig, ax = plt.subplots()
ax.plot(np.arange(ll), phi_list, '-r', np.arange(ll), phi_n, ':b')
plt.xlabel('n')
plt.ylabel("phi^n")
plt.legend(['recursive', 'non-recursive'])
fig.savefig("test_accuracy", dpi=300)

fig, ax = plt.subplots()  
# print(np.arange(ll), np.arange(ll)**(1/2))
ax.plot(np.arange(ll), abs(np.array(phi_n) - np.array(phi_list))) #, np.arange(ll), (math.e)**(0.5*np.arange(ll)-40)
ax.set_yscale('log')
plt.xlabel('n')
plt.ylabel("|phi^n_recursive - phi^n_exact|")
# plt.legend(['recursive', 'non-recursive'])
fig.savefig("test_error", dpi=300)