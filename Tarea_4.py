#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math as m
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


## Defino la funcion a tratar

def f(x):        
    f = 1/(1 + 25*(x**2))
    return f


# In[3]:


## Defino el intervalo y la particion a graficar.

a = -1         
b = 1
p = 0.01

m = np.arange(a,b+p,p)


# In[4]:


## Defino los puntos que voy a tomar y la funcion que interpola con estos.

def g(n):
    g = np.arange(a,b+(b-a)/(2*n) ,(b-a)/n)
    return g

def Lag(x,y,m): 
    
    a = [1]*len(x)
    
    for i in range(len(x)):
        
        for j in range(len(x)):
            
            if j != i:
             
                a[i] *= ( m - x[j])/( x[i] - x[j]) 
                
    return sum(a[t]*y[t] for t in range(len(x)) )
    


# In[10]:


plt.grid()

plt.xlabel('x')
plt.ylabel('y')

plt.plot(m,f(m),'k')
plt.plot(m,Lag(g(6),f(g(6)),m),'y')
plt.plot(m,Lag(g(8),f(g(8)),m),'g')
plt.plot(m,Lag(g(10),f(g(10)),m),'r')
plt.plot(m,Lag(g(12),f(g(12)),m),'b')

plt.legend(('Function','n=6', 'n=8', 'n=10' ,'n=12'),prop = {'size':8}, loc = 'lower center')


# In[ ]:





# In[13]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




