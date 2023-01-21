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
nl=4
x0=0.0; lx=1.0
Lenght= lx-x0

nlh=[8]
dxlh=[(Lenght)/(nlh[0])]

xlh=[np.zeros(int(nlh[0]+1))]
xclh=[np.zeros(int(nlh[0]+2))]
aWlh=[np.zeros(int(nlh[0]))];aElh=[np.zeros(int(nlh[0]))]
aPlh=[np.zeros(int(nlh[0]))];Sulh=[np.zeros(int(nlh[0]))]

rlh=[np.zeros(int(nlh[0])+2)]; 
elh=[np.zeros(int(nlh[0])+2)]; 

T=150*np.ones(nlh[0]+2)
Tlh=[150*np.ones(nlh[0]+2)]

A=0.01; k=5; q=20000; TA=100; TB=500;

tolerance=1e-7; max_iter=10000; count_iter=0; residual=1.0; 
V_aux=[0,0,0]  
    
ResFMG6=[]
TimFMG6=[]
nitFMG6=[]

#ResW=[]
#TimW=[]
#nitW=[]

#ResF=[]
#TimF=[]
#nitF=[]

#ResFMG=[]
#TimFMG=[]
#nitFMG=[]

#Ecuaciones discretizadas
#________________________
def Mallado(xc,x,x0,xl,nx):

    dx=1.0/nx

    for i in range(nx+1):
        x[i]=i*dx
        x[i]=x0+(xl-x0)*x[i]
       
    xc[0]=x[0]
    xc[nx+1]=x[nx]

    for i in range(1,nx+1):
        xc[i]=(x[i]+x[i-1])*0.5
       
    return x,xc

#____________________________________________________________________
def Ecdiscreta(aP,aW,aE,Su,dx,k,A,q,TA,TB,n):
 
    for i in range(n):  
   
        aW[i]=k*A/dx
        aE[i]=k*A/dx
        sp=-2*k*A/dx
             
        if i==0:             #Para el primer nodo
            aW[i]=0
            Su[i]=q*A*dx+2*k*A*TA/dx

        elif i==n-1:            #Para el último nodo
            aE[i]=0
            Su[i]=q*A*dx+2*k*A*TB/dx

        else:   #Para los nodos intermedios
            sp=0
            Su[i]=q*A*dx
           
        aP[i]=aW[i]+aE[i]-sp  
           
    return aP,aW,aE,Su
#____________________________________________________________________
def SOR(aP,aW,aE,Su,nx,T,iter_max):
   
    param=1.0
    T0=np.zeros(nx+2)
  
    for i in range(nx):
        T0[i]=T[i]
                
    iter_GS=0
    
    while iter_GS < iter_max :  
        for i in range(1,nx+1):           
            T[i]=(1.0-param)*T0[i]+param*(Su[i-1]+aW[i-1]*T[i-1]+aE[i-1]*T[i+1])/aP[i-1]
   
        iter_GS=iter_GS+1
         
    return T
#____________________________________________________________________
def Residual(aP,aW,aE,Su,rh,nlh,T):
    
    acum= np.zeros(nlh)
           
    for i in range(1,nlh+1):  #i+1=número de nodo
        acum[i-1]=Su[i-1]-(aP[i-1]*T[i]-aW[i-1]*T[i-1]-aE[i-1]*T[i+1])
        rh[i]=acum[i-1]
             
    residual=sqrt(mean(square(acum)))          
     
    return rh,residual
#____________________________________________________________________
def Restriction(nr):

    for i in range (1,nr):        
    #Definicion de Parámetros    

        r_aux=np.zeros(nlh[i])
       
     #Restricción              
        rlh[i]=Prom(rlh[i-1],nlh[i])
      
        aPlh[i],aWlh[i],aElh[i],Sulh[i]=Ecdiscreta(aWlh[i],aPlh[i],aElh[i],Sulh[i],dxlh[i],k,A,q,TA,TB,nlh[i]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

        for j in range(1,nlh[i]+1):
            r_aux[j-1]=rlh[i][j]

        elh[i]=SOR(aPlh[i],aWlh[i],aElh[i],r_aux,nlh[i],elh[i],10)

        if i!=nl-1:
            for j in range(1,nlh[i]+1):
                rlh[i][j]=rlh[i][j]-(aPlh[i][j-1]*elh[i][j]-aWlh[i][j-1]*elh[i][j-1]-aElh[i][j-1]*elh[i][j+1])
#____________________________________________________________________

def Restriction_FMG(n_r):

    for i in range (1,n_r):    

        r_aux=np.zeros(nlh[i-1])

        #Restricción              
        rlh[i-1]=Prom(rlh[i],nlh[i-1])

        for j in range(1,nlh[i-1]+1):
            r_aux[j-1]=rlh[i][j]

        elh[i-1]=SOR(aPlh[i-1],aWlh[i-1],aElh[i-1],r_aux,nlh[i-1],elh[i-1],10)

        if i!=n_r:
            for j in range(1,nlh[i-1]+1):
                rlh[i-1][j]=rlh[i-1][j]-(aPlh[i-1][j-1]*elh[i-1][j]-aWlh[i-1][j-1]*elh[i-1][j-1]-aElh[i-1][j-1]*elh[i-1][j+1])
                               
#____________________________________________________________________
def Prolongation(n_p):

    for i in range(1,n_p):    

        elhp=np.zeros(nlh[n_p-i-1])
        r_aux=np.zeros(nlh[n_p-i-1])

        elhp=Interpolation(elh[n_p-i],nlh[n_p-i],nlh[n_p-i-1])  
        elh[n_p-i-1]=elh[n_p-i-1]+elhp

        for j in range(1,nlh[n_p-i-1]+1):
            r_aux[j-1]=rlh[n_p-i-1][j]

        elh[n_p-i-1]=SOR(aPlh[n_p-i-1],aWlh[n_p-i-1],aElh[n_p-i-1],r_aux,nlh[n_p-i-1],elh[n_p-i-1],2)

#____________________________________________________________________

def Prolongation_FMG(n_p):

    for i in range(1,n_p):    

        elhp=np.zeros(nlh[i])
        r_aux=np.zeros(nlh[i])

        elhp=Interpolation(elh[i-1],nlh[i-1],nlh[i])  
        elh[i]=elh[i]+elhp

        for j in range(1,nlh[i]+1):
            r_aux[j-1]=rlh[i][j]

        elh[i]=SOR(aPlh[i],aWlh[i],aElh[i],r_aux,nlh[i],elh[i],2)
        
#____________________________________________________________________    
def Prom(rlh,nlh):

    r2lh=np.zeros(nlh+2)
   
    for i in range(1,nlh+1):  
        r2lh[i]=(rlh[2*i]+rlh[2*i-1])/2    
  
    return r2lh
#____________________________________________________________________
def Interpolation(e2lh,n2lh,nlh):    
    
    ehp=np.zeros(nlh+2)
    
    for i in range(1,n2lh+1):      
        ehp[2*i-1]=0.75*e2lh[i]+0.25*e2lh[i-1]
        ehp[2*i]=0.75*e2lh[i]+0.25*e2lh[i+1]
           
    return ehp
#____________________________________________________________________
def V_Cycle(nl):    
    
    Restriction(nl)
    Prolongation(nl)

#____________________________________________________________________
def W_Cycle(nl):    
    
    Restriction(nl)
    Prolongation(1)
    Restriction(1)
    Prolongation(2)
    Restriction(2)
    Prolongation(1)
    Restriction(1)
    Prolongation(nl)

#____________________________________________________________________
def F_Cycle(nl):
    
    Restriction(nl)
    Prolongation(1)
    Restriction(1)
    Prolongation(2)
    Restriction(2)
    Prolongation(nl)

#___________________________________________________________________

def FMG_Cycle(nl):
    
    for i in range(1,nl):                

        Tlh[i]=Interpolation(Tlh[i-1],nlh[i-1],nlh[i])  
        aPlh[i],aWlh[i],aElh[i],Sulh[i]=Ecdiscreta(aPlh[i],aWlh[i],aElh[i],Sulh[i],dxlh[i],k,A,q,TA,TB,nlh[i]) #Se calculan los coeficientes aW,aE,ap,Su,Sp
        Tlh[i]=SOR(aPlh[i],aWlh[i],aElh[i],Sulh[i],nlh[i],Tlh[i],2)  
        rlh[i],residual = Residual(aPlh[i],aWlh[i],aElh[i],Sulh[i],rlh[i],nlh[i],Tlh[i])
    
        Restriction_FMG(i+1)
        Prolongation_FMG(i+1)

        Tlh[i]=Tlh[i]+elh[i]
        Tlh[i]=SOR(aPlh[i],aWlh[i],aElh[i],Sulh[i],nlh[i],Tlh[i],2)  
        rlh[i],residual = Residual(aPlh[i],aWlh[i],aElh[i],Sulh[i],rlh[i],nlh[i],Tlh[i])

"""
    Tlh[nl-1]=Tlh[nl-1]+elh[nl-1]
    Tlh[nl-1]=SOR(aPlh[nl-1],aWlh[nl-1],aElh[nl-1],Sulh[nl-1],nlh[nl-1],Tlh[nl-1],2)  
    rlh[nl-1],residual = Residual(aPlh[nl-1],aWlh[nl-1],aElh[nl-1],Sulh[nl-1],rlh[nl-1],nlh[nl-1],Tlh[nl-1])
""" 
#___________________________________________________________________
def Parametros(nl,a):

    for i in range(1,nl):

        if a==0:        
            nlh.append(int(nlh[i-1]/2))
            dxlh.append((Lenght)/(nlh[i]))

        if a==1:
            nlh.append(int(nlh[i-1]*2))
            dxlh.append((Lenght)/(nlh[i]))
            
            Tlh.append(150*np.ones(nlh[i]+2))

        xlh.append(np.zeros(nlh[i]+1))
        xclh.append(np.zeros(nlh[i]+2))        
        xlh[i],xclh[i]=Mallado(xclh[i], xlh[i], x0, lx, nlh[i])
        
        aPlh.append(np.zeros(int(nlh[i])))
        aWlh.append(np.zeros(int(nlh[i])))
        aElh.append(np.zeros(int(nlh[i])))
        Sulh.append(np.zeros(int(nlh[i])))

        rlh.append(np.zeros(int(nlh[i])+2))
        elh.append(np.zeros(int(nlh[i])+2))                

#____________________________________________________________________
        

def boundaries(T,n,dx,TA,TB):
    T[0]=TA
    T[n+1]=TB
    
    return T

#____________________________________________________________________
def Plot_T(T,x):
 
    plt.plot(x,T,marker ="o")
    plt.xlabel('Distance(m)')
    plt.ylabel('Temperature(Â°C)')
    plt.title('Temperature Distribution')
    plt.show()            

#____________________________________________________________________

def Resverse():

    Tlh.reverse()

    nlh.reverse()
    dxlh.reverse()
    
    xlh.reverse()
    xclh.reverse()
        
    aPlh.reverse()
    aWlh.reverse()
    aElh.reverse()
    Sulh.reverse()

    rlh.reverse()
    elh.reverse()

#____________________________________________________________________

def R_norm(n,r_0):
    
    ResFMG6.append(residual/r_0)
    n=n+1
    nitFMG6.append(n)
    t_par= time.process_time()
    end2=t_par-start
    TimFMG6.append(end2)

    return (n)

#____________________________________________________________________

def Multimalla_VWF(T,residual):
         
    n=0; iter_MM=0
    Parametros(nl,0)    

    while iter_MM < max_iter and residual > tolerance:  

        T=SOR(aPlh[0],aWlh[0],aElh[0],Sulh[0],nlh[0],T,5)  
        rlh[0],residual = Residual(aPlh[0],aWlh[0],aElh[0],Sulh[0],rlh[0],nlh[0],T)

        if n==0:
            r_0=residual
            n=R_norm(n,r_0)
    
        V_Cycle(nl)
#        W_Cycle(nl)
#        F_Cycle(nl)

        T=T+elh[0]
        T=SOR(aPlh[0],aWlh[0],aElh[0],Sulh[0],nlh[0],T,2)  
        rlh[0],residual = Residual(aPlh[0],aWlh[0],aElh[0],Sulh[0],rlh[0],nlh[0],T)

        n=R_norm(n,r_0)
      
    T=boundaries(T,nlh[0],dxlh[0],TA,TB)

    return(T)
#____________________________________________________________________

def Multimalla_FMG(T,residual):

    n=0; iter_MM=0
    Parametros(nl,1)    

    Tlh[0]=SOR(aPlh[0],aWlh[0],aElh[0],Sulh[0],nlh[0],Tlh[0],10)  
    rlh[0], residual = Residual(aPlh[0],aWlh[0],aElh[0],Sulh[0],rlh[0],nlh[0],Tlh[0])

    r_0=residual
    n=R_norm(n,r_0)

    FMG_Cycle(nl)

    for i in range (nl):
        elh[i]=np.zeros(nlh[i]+2)

    Resverse()

    rlh[0], residual = Residual(aPlh[0],aWlh[0],aElh[0],Sulh[0],rlh[0],nlh[0],Tlh[0])

    n=R_norm(n,r_0)

    while iter_MM < max_iter and residual > tolerance:  

        Restriction(nl)
        Prolongation(nl)

        Tlh[0]=Tlh[0]+elh[0]
        Tlh[0]=SOR(aPlh[0],aWlh[0],aElh[0],Sulh[0],nlh[0],Tlh[0],2)  
        rlh[0],residual = Residual(aPlh[0],aWlh[0],aElh[0],Sulh[0],rlh[0],nlh[0],Tlh[0])
        n = R_norm(n,r_0)
    
        iter_MM = iter_MM+1     

        for i in range (nl):
            elh[i]=np.zeros(nlh[i]+2)

    Tlh[0]=boundaries(Tlh[0],nlh[0],dxlh[0],TA,TB)

    return(Tlh[0])
#____________________________________________________________________
#Programa principal

xlh[0],xclh[0] = Mallado(xclh[0], xlh[0], x0, lx, nlh[0])
aPlh[0],aWlh[0],aElh[0],Sulh[0]=Ecdiscreta(aPlh[0],aWlh[0],aElh[0],Sulh[0],dxlh[0],k,A,q,TA,TB,nlh[0]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

#T = Multimalla_VWF(T,residual)
#plt.figure(1)
#Plot_T(T,xclh[0])

Tlh[0] = Multimalla_FMG(Tlh[0],residual)
plt.figure(1)
Plot_T(Tlh[0],xclh[0])

#plt.figure(2)
#plt.subplot(221)
#plt.plot(nitV,ResV)
#plt.subplot(222)
#plt.plot(nitV,TimV)
#plt.subplot(223)
#plt.plot(TimV,ResV)

#print(T)
#print(residual)
#print(iter_MM)
end= time.process_time()
print(end-start)
