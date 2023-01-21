# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import time
start= time.process_time()

from numpy import mean, sqrt, square
import math
import numpy as np
import matplotlib.pyplot as plt

#DefiniciÃ³n de variables

dx=0.05; Lenght=1.0; n=int(Lenght/dx+2) #NÃºmero de nodos
x=np.zeros(n)
TB=100; Tamb=20; hPkA=25 #m^2
aP=np.zeros(n); aW=np.zeros(n); aE=np.zeros(n); Su=np.zeros(n)
A=np.zeros(n); Cp=np.zeros(n);
T=np.zeros(n)
rh=np.zeros(n); tolerance=1e-7

Res=[]
Tim=[]
nit=[]

#Ecuaciones discretizadas
#____________________________________________________________________
def Ecdiscreta(dx,hpkA,Tamb,TB,n):
       
    for i in range(1,n-1): 
    
        aW[i]=1/dx
        aE[i]=1/dx
        Sp=-hPkA*dx
         
        if i==1:                #Para el primer nodo
            aW[i]=0
            Sp=Sp-2/dx
            Su[i]=hPkA*dx*Tamb+2*TB/dx            

        elif i==n-2:            #Para el Ãºltimo nodo
            aE[i]=0
            Su[i]=hPkA*dx*Tamb

        else:                    #Para los nodos intermedios
            Su[i]=hPkA*dx*Tamb
        
        aP[i]=aW[i]+aE[i]-Sp  
        
    return aW,aP,aE,Su
#____________________________________________________________________
def Thomas(T,aP,aW,aE,Su,n,Cp,A,TB,tolerance):

    max_iter=1000; residual=1; iter_GS=0; m=0; r_0=0
    
    while iter_GS < max_iter and residual > tolerance:  

       TA=0
       Cp[0]=TB
       Cp[n-1]=T[n-1]

       for i in range(1,n-1):
              A[i]=aE[i]/(aP[i]-(aW[i]*A[i-1]))
              Cp[i]=(aW[i]*Cp[i-1]+Su[i])/(aP[i]-aW[i]*A[i-1])     
  
       T[n-1]=A[n-1]*TA+Cp[n-1]
          
       for k in range(n-2,-1,-1):
              T[k]=A[k]*T[k+1]+Cp[k]

       iter_GS=iter_GS+1 
          
       for i in range(1,n-1):  #i+1=número de nodo
              rh[i]=Su[i]+aW[i]*T[i-1]+aE[i]*T[i+1]-aP[i]*T[i]
       
       residual=sqrt(mean(square(rh)))          

       if m==0:
           r_0=residual

       Res.append(residual/r_0)
       m=m+1
       nit.append(m)   
       t_par= time.process_time()
       end2=t_par-start
       Tim.append(end2)
     
    return T,residual,iter_GS

#____________________________________________________________________
def GaussSeidel(aW,aP,aE,Su,n,T,tolerance):
    
    max_iter=1000; residual=1; iter_GS=0; m=0; r_0=0
    
    while iter_GS < max_iter and residual > tolerance:  
        for i in range(1,n-1):
           
            T[i]=(Su[i]+aW[i]*T[i-1]+aE[i]*T[i+1])/aP[i]
    
        iter_GS=iter_GS+1 
          
        for i in range(1,n-1):  #i+1=número de nodo
            rh[i]=Su[i]+aW[i]*T[i-1]+aE[i]*T[i+1]-aP[i]*T[i]

        residual=sqrt(mean(square(rh)))          

        if m==0:
            r_0=residual 

        Res.append(residual/r_0)
        m=m+1
        nit.append(m)   
        t_par= time.process_time()
        end2=t_par-start
        Tim.append(end2)
     
    return T,residual,iter_GS
#____________________________________________________________________
def SOR(aW,aP,aE,Su,n,T,tolerance):
    
    T0=np.zeros(n)
    max_iter=1000; residual=1; iter_GS=0; m=0; r_0=0
    param=1.2

    for i in range(n):
        T0[i]=T[i]

    while iter_GS < max_iter and residual > tolerance:  
        for i in range(1,n-1):
           
            T[i]=(1.0-param)*T0[i]+param*(Su[i]+aW[i]*T[i-1]+aE[i]*T[i+1])/aP[i]
    
        iter_GS=iter_GS+1 

        for i in range(1,n-1):  #i+1=número de nodo
            rh[i]=Su[i]+aW[i]*T[i-1]+aE[i]*T[i+1]-aP[i]*T[i]

        residual=sqrt(mean(square(rh)))   

        if m==0:
            r_0=residual

        Res.append(residual/r_0)
        m=m+1
        nit.append(m)   
        t_par= time.process_time()
        end2=t_par-start
        Tim.append(end2)
         
    return T,residual,iter_GS
#____________________________________________________________________
def boundaries(T,n,dx,Tamb,TB):
    T[0]=TB
    T[n-1]=T[n-2]
    
    return T

#____________________________________________________________________
def Plot_T(T,x,dx,Lenght,n):
    x[1]=dx/2
    x[n-1]=Lenght
    for i in range(2,n-1):  
        x[i]=x[i-1]+dx
 
    plt.plot(x,T,marker ="o")
    plt.xlabel('Distance(m)')
    plt.ylabel('Temperature(Â°C)')
    plt.title('Temperature Distribution')
    plt.show()            
             
#____________________________________________________________________

#Programa principal

aW,aP,aE,Su = Ecdiscreta(dx,hPkA,Tamb,TB,n) #Se calculan los coeficientes aW,aE,ap,Su,Sp
#T,residual,iter_GS=Thomas(T,aP,aW,aE,Su,n,Cp,A,TB,tolerance)    #Se resuelve la matriz Tridiagonal por el algoritmo de Thomas
#T,residual,iter_GS=GaussSeidel(aW,aP,aE,Su,n,T,tolerance)   #Se resuelve la matriz Tridiagonal por el algoritmo de Thomas
T,residual,iter_GS=SOR(aW,aP,aE,Su,n,T,tolerance)   

T=boundaries(T,n,dx,Tamb,TB)

plt.figure(1)
Plot_T(T,x,dx,Lenght,n)

plt.figure(2)
#plt.subplot(221)
plt.plot(nit,Res,marker ="o")
#plt.subplot(222)
#plt.plot(nit,Tim)
#plt.subplot(223)
#plt.plot(Tim,Res)

#print(T)
#print(residual)
#print(iter_GS)
end= time.process_time()
print(end-start)
