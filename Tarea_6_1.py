#!/usr/bin/env python
# coding: utf-8

# In[29]:


import math as m
import numpy as np
import matplotlib.pyplot as plt
from math import *


# In[30]:


a = 0
b = m.pi

def g(n):
    g = np.arange(a,b+(b-a)/(2*n) ,(b-a)/n)
    return g

print(g(3))

def f(x):
    f = np.sin(x)
    return f


# The Midpoint Rule

# In[31]:


def m_r(x,t):
    
    s = 0
    for i in range (len(x)-1):
        s += (x[i+1]-x[i])*t((x[i]+x[i+1])/2)
    print("The Midpoint Rule para",len(x)-1,"intervalos La integral converge a =",s)

m_r(g(10),f)
m_r(g(50),f)
m_r(g(100),f)


# The Trapezoidal Rule

# In[32]:


def t_r(x,t):
    
    s = 0
    for i in range (len(x)-1):
        s += (x[i+1]-x[i])*(t(x[i+1])+t(x[i]))/2
    print("The Trapezoidal Rule para ",len(x)-1, "La integral converge a =",s)

t_r(g(10),f)
t_r(g(50),f)
t_r(g(100),f)


# The Simpson Rule

# In[33]:


def s_r(x,t):
    
    s = 0
    for i in range (len(x)-1):
        s += ((x[i+1]-x[i])/6)*(t(x[i])+4*t((x[i]+x[i+1])/2)+t(x[i+1]))
    print("The Simpson Rule para",len(x)-1,"intervalos La integral converge a =",s)

s_r(g(10),f)
s_r(g(50),f)
s_r(g(100),f)


# $ f(x) = xsin(x)\              Valor\ Teorico\ de\ la\ Integral\ es\ \pi $

# In[34]:


def F(x):
    F = x*np.sin(x)
    return F


# In[35]:


m_r(g(10),F)
t_r(g(10),F)
s_r(g(10),F)

m_r(g(50),F)
t_r(g(50),F)
s_r(g(50),F)

m_r(g(100),F)
t_r(g(100),F)
s_r(g(100),F)


# In[ ]:




