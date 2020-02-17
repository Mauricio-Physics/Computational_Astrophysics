#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math as m
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


d = 1e-8
n = 0.0

def F(x):
    F = x - np.sin(x)
    return F

def D_F(x):
    D_F = 1 - np.cos(x)
    return D_F

def D_f(x,d):
    D_f= (F(x+d) - F(x))/d
    return D_f
print("First order forward diference =",D_f(n,d))

def D_b(x,d):
    D_b = (F(x) - F(x-d))/d
    return D_b
print("First order backward diference =", D_b(n,d))

def D_c(x,d):
    D_c = (F(x+d) - F(x-d))/(2*Delta)
    return D_c
print("Second order central diference =",D_c(n,d))


# In[4]:


x = np.arange(-1,1.01,0.01)
f = F(x)
DF = D_F(x)
Df = D_f(x,d)
Db = D_b(x,d)
Dc = D_c(x,d)

plt.grid()
plt.plot(x,f,'b')
plt.plot(x,Df,'k')
plt.plot(x,Db,'g')
plt.plot(x,Dc,'y')
plt.legend(('Function', 'First order forward diference ', 'First order backward diference', 'Second order central diference'),prop = {'size':10}, loc = 'lower center')


# In[10]:


plt.grid()
plt.plot(x,DF,'r')

plt.grid()
plt.plot(x,Df,'k')
plt.plot(x,Db,'g')
plt.plot(x,Dc,'y')

plt.legend(('Analytic derivative', 'First order forward diference ', 'First order backward diference', 'Second order central diference'),prop = {'size':10}, loc = 'lower center')


# 

# Segunda Derivada

# In[20]:


y = m.pi/2

def D_2F(x):
    D_2F =  np.sin(x)
    return D_2F

z = D_2F(x)

def D_2f(x,d):
    D_2f = (F(x+d)+F(x-d)-2*F(x))/(d**2)
    return D_2f

r = D_2f(x,d)
q = D_2f(x,1e-7)
w = D_2f(x,1e-6)

print("Second Derivative of the function =",D_2f(n,d))
print("Second Derivative of the function =",D_2f(n,d))


plt.grid()
plt.plot(x,z,'r')
plt.plot(x,r,'b')
plt.plot(x,q,'g')
plt.plot(x,w,'k')
plt.legend(('Second Analytical Derivative', 'Numerical with Delta = 1e-8','Numerical with Delta = 1e-7','Numerical with Delta = 1e-6'),prop = {'size':10}, loc = 'lower center')


# In[ ]:




