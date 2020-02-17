#!/usr/bin/env python
# coding: utf-8

# In[15]:


import math as m
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


eps = 1e-8
T = 365.25635
w = 2*m.pi/T
d = 1e-8

a = 1
b = 4
x = np.arange(a,b,1)
t = float(input("ingrese el valor de t:"))
e = float(input("ingrese el valor de e:"))

def f1(x):
    f1 = x - w*t - e*np.sin(x)
    return f1

plt.grid()

plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,f1(x),'k')


# In[3]:


def f(z):
    f = z - w*t - e*m.sin(z)
    return f


def d_f(s):
    d_f = (f(s+d) - f(s-d))/(2*d)
    return d_f


# Newthon Raphson

# In[22]:


def n_r(xo):
    y = xo
    i = 0
    a = 0
    while abs(f(y)) > eps:
         i += 1 
         y = y - f(y)/d_f(y)
    print("La raiz converge a",y,"despues de",i,"iteraciones")
    print("Para t =",t,"y una excentricidad =",e," x (AU) =", m.cos(y))
    
    
n_r(2.5)
            


# Secante

# In[23]:


def s(xo,x1):
    tol = 1e-10
    z = xo
    y = x1
    j = 0
    x = 0
    while abs((y-z)/y) > tol:
        j += 1
        x = y - f(y)*(y-z)/(f(y)-f(z))
        z = y
        y = x
    print("La raiz converge a",y,"despues de",j,"iteraciones")    
    print("Para t =",t,"y una excentricidad =",e," x (AU) =", m.cos(y))
    
s(2,2.5)


# Biseccion

# In[24]:


def b(xo,x1):
    tol = 1e-10
    a = xo
    b = x1
    k  = 0
    
    while b-a > tol:
          k += 1
          c = (a+b)/2
            
          if f(a)*f(c) < 0:
             b = c
          else:
             a = c
                
    print("La raiz converge a",c,"despues de",k,"iteraciones") 
    print("Para t =",t,"y una excentricidad =",e," x (AU) =", m.cos(c))
        
b(2,2.55)


# In[ ]:




