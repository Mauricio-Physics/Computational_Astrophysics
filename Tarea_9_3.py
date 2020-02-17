#!/usr/bin/env python
# coding: utf-8

# In[13]:


import math as m
import numpy as np
import matplotlib.pyplot as plt
from math import *


# In[14]:


a_1 =np.loadtxt("3.a.dat",unpack=True)
b_1 =np.loadtxt("3.b.dat",unpack=True)
r = b_1.size
x = [0]*r


# In[15]:


def e_g(a,b):                                          ## Aqui defino el codigo de la eliminacion gaussiana
    for i in range(0,b.size):
        for j in range(i+1,b.size):
            b[j]= ((-a[j,i]/a[i,i]) * b[i] + b[j])
            a[j,:]= ((-a[j,i]/a[i,i]) * a[i,:] + a[j,:])  
    for k in range(2,r+1):
        x[r-1] = b_1[r-1]/a_1[r-1][r-1]
        x[r-k] = (b_1[r-k]-np.dot(a_1[r-k,:],x))/a_1[r-k][r-k]
    return x
    print(x)
        
e_g(a_1,b_1)


# In[16]:


def det(a,b):                                                     ## Aqui llamo los modulos del determinante y del solucionador
    if np.linalg.det(a) != 0:
        x = np.linalg.solve(a,b)
        print("El sistema tiene solucion su determinante es",np.linalg.det(a))
        print("Su solucion es",x)
    else:
        print("el sistema no tiene solucion")
    return x
det(a_1,b_1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




