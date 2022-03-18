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

#Definición de variables
nl=3

dxlh=[0.05]; Lenght=1.0 #m
nlh=[math.ceil(Lenght/dxlh[0]+2)]
x=np.zeros(int(nlh[0]))
aWlh=[np.zeros(int(nlh[0]))];aPlh=[np.zeros(int(nlh[0]))];aElh=[np.zeros(int(nlh[0]))];Sulh=[np.zeros(int(nlh[0]))]
rlh=[np.zeros(int(nlh[0]))]; elh=[np.zeros(int(nlh[0]))]; elhp=[np.zeros(int(nlh[0]))]        
rh=np.zeros(int(nlh[0]))
T=150*np.ones(nlh[0])

A=0.01; k=5; q=20000; TA=100; TB=500;

tolerance=1e-7; max_iter=10000; count_iter=0; residual=1.0; iter_GS=0

#Ecuaciones discretizadas
#____________________________________________________________________
def Ecdiscreta(aW, aP, aE, Su,dx,k,A,q,TA,TB,n):
 
    for i in range(1,n-1):  
   
        aW[i]=k*A/dx
        aE[i]=k*A/dx
        sp=-2*k*A/dx
             
        if i==1:             #Para el primer nodo
            aW[i]=0
            Su[i]=q*A*dx+2*k*A*TA/dx

        elif i==n-2:            #Para el último nodo
            aE[i]=0
            Su[i]=q*A*dx+2*k*A*TB/dx

        else:   #Para los nodos intermedios
            sp=0
            Su[i]=q*A*dx
           
        aP[i]=aW[i]+aE[i]-sp  
           
    return aW,aP,aE,Su
#____________________________________________________________________
def SOR(aW,aP,aE,Su,n,T,iter_max):
   
    param=1.2
    T0=T
 
    iter_GS=0
    while iter_GS < iter_max :  
        for i in range(1,n-1):
           
            T[i]=(1.0-param)*T0[i]+param*(Su[i]+aW[i]*T[i-1]+aE[i]*T[i+1])/aP[i]
   
        iter_GS=iter_GS+1
         
    return T
#____________________________________________________________________
def Residual(aW,aP,aE,Su,nmh,T):

    rh=np.zeros(nmh)
    for i in range(1,nmh-1):  #i+1=número de nodo
        rh[i]=Su[i]-(-aW[i]*T[i-1]+aP[i]*T[i]-aE[i]*T[i+1])
        residual=sqrt(mean(square(rh)))          
     
    return rh,residual
#____________________________________________________________________
def Restriction(nr):

    for i in range (1,nr):        
    #Definicion de Parámetros    
        if iter_GS==0:
            Parametros(i)
       
     #Restricción              
        rlh[i],elh[i]=Prom(rlh[i-1],nlh[i],aWlh[i],aPlh[i],aElh[i],Sulh[i],dxlh[i],k,A,q,TA,TB)
        if i!=nl-1:
            for j in range(1,nlh[i]-1):
                rlh[i][j]=rlh[i][j]+aWlh[i][j]*elh[i][j-1]+aElh[i][j]*elh[i][j+1]-aPlh[i][j]*elh[i][j]

#___________________________________________________________________
def Parametros(i):

    dxlh.append((2**i)*dxlh[0])
    nlh.append(math.ceil(Lenght/dxlh[i])+2)
    aWlh.append(np.zeros(int(nlh[i])))
    aPlh.append(np.zeros(int(nlh[i])))
    aElh.append(np.zeros(int(nlh[i])))
    Sulh.append(np.zeros(int(nlh[i])))
    rlh.append(0)
    elh.append(0)            
    elhp.append(0)
#____________________________________________________________________    
def Prom(rmh,nmh,aW2h,aP2h,aE2h,Su2h,dx2h,k,A,q,TA,TB):
   
    r2mh=np.zeros(nmh)
    e2mh=np.zeros(nmh)
   
    for i in range(1,nmh-1):  
        r2mh[i]=(rmh[2*i]+rmh[2*i-1])/2    
 
    aW2h,aP2h,aE2h,Su2h=Ecdiscreta(aW2h,aP2h,aE2h,Su2h,dx2h,k,A,q,TA,TB,nmh) #Se calculan los coeficientes aW,aE,ap,Su,Sp
    e2mh=SOR(aW2h,aP2h,aE2h,r2mh,nmh,e2mh,10)
 
    return r2mh,e2mh
#____________________________________________________________________
def Prolongation(np):

    for i in range(1,nl):    
        elhp[i]=Interpolation(elh[nl-i],nlh[nl-i],nlh[nl-i-1])  
        elh[nl-i-1]=elh[nl-i-1]+elhp[i]
#____________________________________________________________________
def Interpolation(e2h,n2h,nh):    
 
    ehp=np.zeros(nh)
    for i in range(1,n2h-1):      
        ehp[2*i-1]=0.75*e2h[i]+0.25*e2h[i-1]
        ehp[2*i]=0.75*e2h[i]+0.25*e2h[i+1]
           
    return ehp
#____________________________________________________________________
def V_Cycle():    
 
    Restriction(3)
    Prolongation(3)

#____________________________________________________________________
def W_Cycle():    
 
    Restriction(3)
    Prolongation(1)
    Restriction(1)
    Prolongation(2)
    Restriction(2)
    Prolongation(1)
    Restriction(1)
    Prolongation(3)
#____________________________________________________________________
def F_Cycle():

    Restriction(3)
    Prolongation(1)
    Restriction(1)
    Prolongation(2)
    Restriction(2)
    Prolongation(3)
#____________________________________________________________________

def boundaries(T,n,dx,TA,TB):
    T[0]=TA
    T[n-1]=TB
    
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

aWlh[0],aPlh[0],aElh[0],Sulh[0]=Ecdiscreta(aWlh[0],aPlh[0],aElh[0],Sulh[0],dxlh[0],k,A,q,TA,TB,nlh[0]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

while iter_GS < max_iter and residual > tolerance:  
       
    T=SOR(aWlh[0],aPlh[0],aElh[0],Sulh[0],nlh[0],T,5)  
    rlh[0],residual=Residual(aWlh[0],aPlh[0],aElh[0],Sulh[0],nlh[0],T)

    F_Cycle()

    T=T+elhp[nl-1]
    T=SOR(aWlh[0],aPlh[0],aElh[0],Sulh[0],nlh[0],T,2)  
    iter_GS=iter_GS+1     
    rlh[0],residual=Residual(aWlh[0],aPlh[0],aElh[0],Sulh[0],nlh[0],T)
    
T=boundaries(T,nlh[0],dxlh[0],TA,TB)
Plot_T(T,x,dxlh[0],Lenght,nlh[0])

print(T)
print(residual)
print(iter_GS)
end= time.process_time()
print(end-start)
