import torch
import time
import numpy as np


nn = 10000 #trials
tms_qu = []
tms_qu2 = []
tms_sum = []
device = 'cuda:0'

for n in range(nn):
    t = 1*torch.rand(100).to(device)
    # print(t)

    tic = time.time()
    torch.quantile(t, torch.tensor([0.25, 0.5]).to(device))
    # torch.max(t)
    toc = time.time()
    # print('quantile', toc - tic)
    tms_qu.append(toc-tic)

    tic = time.time()
    torch.quantile(t, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).to(device))
    # torch.max(t)
    toc = time.time()
    # print('quantile', toc - tic)
    tms_qu2.append(toc-tic)

    tic = time.time()
    torch.sum(t)
    toc = time.time()
    # print('sum', toc - tic)
    tms_sum.append(toc-tic)

m_qu, m_qu2, m_sum = np.mean(tms_qu), np.mean(tms_qu2), np.mean(tms_sum)
std_qu, std_qu2, std_sum = np.std(tms_qu), np.std(tms_qu2), np.std(tms_sum)
print('qu', m_qu, 'qu2', m_qu2, 'sum', m_sum, 'ratio', m_qu/m_sum, m_qu2/m_sum)
print('errors', std_qu, std_qu2, std_sum)


# qu 0.00010163166761398316 sum 3.702123165130615e-06 ratio 27.4522653841522
# qu 0.00012834925651550292 sum 9.125137329101563e-06 ratio 14.065460265039086 #cuda