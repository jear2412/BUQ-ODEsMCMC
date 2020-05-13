#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:47:01 2020

@author: Javier Aguilar


Gibbs sampling for X ~N_2 ( \mu, \Sigma  )
"""


import numpy as np
import scipy.stats
import seaborn as sns
from timeit import default_timer as timer
import matplotlib.pyplot as plt


#--------- Random scan Gibbs sampling functions

sns.set_style("whitegrid")
 

def chooseK(w,nk): #returns distribution to be sampled of mixture distribution
    #number of kernels is nk
    parts=np.arange(nk)
    randD = scipy.stats.rv_discrete(a=0, b=parts[nk-1],name='randD', values=(parts, w))
    return randD.rvs(size=1)[0]


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X-mux
    Ymu = Y-muy
    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom



def u1(x1,x2,mu,cov):
    #energy function u=-logf
    x=np.array([x1,x2])
    val=-np.log(2*np.pi)-1/2*np.log( np.linalg.det(cov))
    val=val-1/2*( np.transpose(x-mu) )@(np.linalg.inv(cov))@(x-mu )
    u=-val
    return u


#random scan gibbs

def gibbs1(mu, cov,T=10000, w1=1/2, w2=1/2): 
    ''' mu and cov are the parameters of the target distribution (N_2 )
        w_1 and w_2 are weights associated to kernels
    '''
    x1s=np.zeros(T)
    x2s=np.zeros(T)
    u=np.zeros(T)
    x10=scipy.stats.norm.rvs(loc=mu[0],scale= cov[0,0],size=1)   
    x20=scipy.stats.norm.rvs(loc=mu[1],scale= cov[1,1],size=1)
    x1s[0]=x10
    x2s[0]=x20 
    w=np.array([w1,w2])
    u[0]=u1( x1s[0], x2s[0],mu, cov   )
    for i in range(1,T):
        s=chooseK(w,nk=2)
        if(s==0):
            x1p=scipy.stats.norm.rvs(loc=mu[0]+cov[0,1]*( np.sqrt(cov[0,0]/cov[1,1]) )*(x2s[i-1]-mu[1]),scale= np.sqrt( cov[0,0]*(1-cov[0,1]**2)),size=1)   
            x2p=x2s[i-1]
            x1s[i]=x1p
            x2s[i]=x2p      
            u[i]=u1( x1s[i], x2s[i],mu, cov   )
        else:
            x1p=x1s[i-1]
            x2p=scipy.stats.norm.rvs(loc=mu[1]+cov[0,1]*( np.sqrt(cov[1,1]/cov[0,0]) )*(x1s[i-1]-mu[0]),scale= np.sqrt( cov[1,1]*(1-cov[0,1]**2)),size=1)   
            x1s[i]=x1p
            x2s[i]=x2p   
            u[i]=u1( x1s[i], x2s[i],mu, cov   )

    return x1s, x2s, u


def trace_plot(x):
    plt.plot( np.arange(len(x)), x )
    plt.title(' Traceplot ')
    plt.xlabel(r't')
    plt.ylabel(r'$\theta$')
    

#--------- Simulation

mu=np.array([5,3])
rho= -0.90
Sigma=np.array([ [ 1,rho], [rho,1]]) 
#notice the covariance matrix is in terms of the correlation
#rho determines how inclined and thin the distribution is
#try it out with rho= 0.99 and T very big int(1e6)
    

T=10000 #iterations

start=timer()
x1s,x2s,u= gibbs1(mu,Sigma,T)
end=timer()
time=end-start
print('Time taken: ', time)

bi=int(0.1*T)

np.mean(x1s[bi:])
np.mean(x2s[bi:])

np.var(x1s[bi:])
np.var(x2s[bi:])
np.cov(x1s[bi:], x2s[bi:])


trace_plot(x1s)
trace_plot(x2s)

sns.distplot( x1s[bi:]  , kde=False , bins=15).set_title(' Histogram '+r'$x_1$'+' with '+ r' $\rho = $'+str(rho) )
sns.distplot( x2s[bi:]  , kde=False , bins=15).set_title(' Histogram '+r'$x_1$'+' with '+ r' $\rho = $'+str(rho) ) 


tot=len(x1s)
m=1000
X1s =np.linspace(0, 10, m)
X2s= np.linspace(0, 6, m)
X1S, X2S = np.meshgrid(X1s, X2s)
Z=bivariate_normal(  X1S,X2S,mux=mu[0], muy=mu[1],sigmax=Sigma[0,0], sigmay=Sigma[1,1], sigmaxy=rho )
plt.contour(X1s, X2s, Z)
plt.plot(x1s, x2s,'.' ,zorder=-1, )
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.title( 'Contour plot with chains '+r'$\rho=$'+str(rho) )


plt.plot( x1s[0:100], x2s[0:100] ) 
plt.plot( x1s[0:1000], x2s[0:1000] )









