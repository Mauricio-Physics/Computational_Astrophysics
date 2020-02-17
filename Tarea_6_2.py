#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math as m
import numpy as np
import matplotlib.pyplot as plt


# $ Parte\ a\ $

# In[123]:


a = 0
b = 100
d = 1e-8

x = np.arange(a,b,0.1)

def f(x):    ## Defino la funcion
    f = x**2/(np.exp(x)+1)
    return f

y = f(x)

def max(y):       ## Defino el maximo valor que toma
    j = np.argmax(y)
    k = int(j)
    return k

def lim(x,y):  ## Defino donde cortarla
    lim = 0
    for i in range(max(y),len(y)):
        if y[i] < 1e-8 :
            break
    return x[i]

x_1 = np.arange(a,lim(x,y),0.1)   ## Defino el nuevo intervalo

def s_r(x,t):    ## Integro por simpson
    
    s = 0
    for i in range (len(x)-1):
        s += ((x[i+1]-x[i])/6)*(t(x[i])+4*t((x[i]+x[i+1])/2)+t(x[i+1]))
    print("The Simpson Rule para",len(x)-1,"intervalos La integral converge a =",s)

s_r(x_1,f)

plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_1,f(x_1),'b')


# $ Parte\ b\ $

# $ \sum_{i=0} \left(\frac{dn_{e}}{dE}\right)_{i}  * \Delta E $

# In[121]:


def x_2(n):
    x_2 = np.arange(a,lim(x,y) + (lim(x,y)-a)/n ,(lim(x,y)-a)/n )  
    return(x_2)


# In[125]:


def s_r(x,t):    ## Integro por simpson
    
    s = 0
    for i in range (len(x)-1):
        s += ((x[i+1]-x[i])/6)*(t(x[i])+4*t((x[i]+x[i+1])/2)+t(x[i+1]))
    print("La integral para",len(x)-1,"intervalos de energia converge a =",s)

s_r(x_2(5),f)
s_r(x_2(10),f)


# In[ ]:




