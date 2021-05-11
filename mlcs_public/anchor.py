import numpy as np 
import os, csv, glob, sys
import matplotlib.pyplot as plt


w=np.empty(261, dtype='int')
h=np.empty(261, dtype='int')


r=open('train.txt', mode='rt')

lines=r.readlines()
i=0
for line in lines:
    line=line.strip()
    split=line.split(',')
    w[i]=split[3]
    h[i]=split[4]
    i+=1

#print(w)
#print(h)
mean_w=np.mean(w)
mean_h=np.mean(h)
print(mean_w)
print(mean_h)
ratio=np.empty(261,dtype=float)

for i in range(261):
    ratio[i]=w[i]/h[i]

plt.hist(ratio)
plt.show()

plt.hist(w)
plt.show()
plt.hist(h)
plt.show()