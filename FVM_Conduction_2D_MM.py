# -- coding: utf-8 --
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
x0=0.0; lx=1
y0=0.0; ly=1                                       #Longitu
z=0.01

Lenght_x = lx-x0
Lenght_y = ly-y0

nxlh=[16]; nylh=[16]
dxlh=[(lx-x0)/(nxlh[0])]; dylh=[(ly-y0)/(nylh[0])]
Aew=[dylh[0]*z]; Asn=[dxlh[0]*z]


xlh=[np.zeros(nxlh[0]+1)]; ylh=[np.zeros(nylh[0]+1)]
xclh=[np.zeros(nxlh[0]+2)]; yclh=[np.zeros(nylh[0]+2)]

aWlh=[np.zeros([nxlh[0],nylh[0]])]; aElh=[np.zeros([nxlh[0],nylh[0]])]
aSlh=[np.zeros([nxlh[0],nylh[0]])]; aNlh=[np.zeros([nxlh[0],nylh[0]])]
aPlh=[np.zeros([nxlh[0],nylh[0]])]; Splh=[np.zeros([nxlh[0],nylh[0]])]
Sulh=[np.zeros([nxlh[0],nylh[0]])]

rlh=[np.zeros([nxlh[0]+2,nylh[0]+2])]
elh=[np.zeros([nxlh[0]+2,nylh[0]+2])]

N=0      

T=np.zeros([nxlh[0]+2,nylh[0]+2])
Tlh=[np.zeros([nxlh[0]+2,nylh[0]+2])]

Tos=0; Ton=100; Toe=0; flux=500000;                          #Condiciones de Frontera
kT=1000; 

tolerance=1e-7; max_iter=10000; count_iter=0; residual=1.0; iter_MM=0

n=0      
VRes2=[]
VTim2=[]
Vnit2=[]

#Ecuaciones discretizadas
#________________________
def Ecdiscreta(aP,aS,aN,aW,aE,Su,dx,dy,z,Aew,Asn,k,Tos,Ton,flux,Toe,nx,ny):
 
    for i in range(nx):             #i+1 columna del nodo
        for j in range(ny):         #j+1 fila de nodo

            Sp=0                        #Nodos Centrales          
            Su[i,j]=0      
            aW[i,j]=k*Aew/dx  
            aE[i,j]=k*Aew/dx
            aS[i,j]=k*Asn/dy  
            aN[i,j]=k*Asn/dy          
           
            bW=flux*dx*z                #Fuentes de calor en las fronteras
            bE=2*k*Aew*Toe/dx
            bS=2*k*Asn*Tos/dy
            bN=2*k*Asn*Ton/dy
         
            if j==0:                    #Frontera sur
                aS[i,j]=0
                Su[i,j]=2*k*Asn*Tos/dy
               
            if j==ny-1:                 #Frontera norte
                aN[i,j]=0
                Sp=-2*k*Asn/dy
                Su[i,j]=2*k*Asn*Ton/dy
           
            if i==0:               #Frontera oeste
              aW[i,j]=0

              if j==0:                #Borde inferior izquierdo
                  Su[i,j]=bW+bS
               
              elif j==ny-1:           #Borde superior izquierdo
                  Sp=-2*k*Asn/dy
                  Su[i,j]=bW+bN
             
              else:                   #Borde izquierdo
                    Su[i,j]=bW
                           
            elif i==nx-1:         #Frontera este
                aE[i,j]=0
                               
                if j==0:                #Borde inferior derecho
                    Su[i,j]=bE+bS
               
                elif j==ny-1:           #Borde superior derecho
                    Sp=-2*k*Asn/dy
                    Su[i,j]=bE+bN
               
                else:                   #Borde derecho
                    Su[i,j]=bE  

            aP[i,j]=aW[i,j]+aE[i,j]+aS[i,j]+aN[i,j]-Sp
       
    return  aP, aS, aN, aW, aE, Su
#________________________
def SOR(aP,aS,aN,aW,aE,Su,nx,ny,f,max_iter):
   
    param=1.2
    tolerance=1e-7; count_iter=0; residual=1.0        #Criterios de Convergencia
    f0=np.zeros([nx+2,ny+2])

    while count_iter <= max_iter:  
       
        for i in range(nx+2):
            for jj in range(ny+2):
                f0[i,jj]=f[i,jj]

        for i in range(1,nx+1):
            for j in range(1,ny+1):        
                f[i,j]=(1.0-param)*f0[i,j]+param*(Su[i-1,j-1]+aW[i-1,j-1]*f[i-1,j]+aE[i-1,j-1]*f[i+1,j]+aS[i-1,j-1]*f[i,j-1]+aN[i-1,j-1]*f[i,j+1])/aP[i-1,j-1]
 
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo

    return f
#________________________
def Residual(aPlh,aSlh,aNlh,aWlh,aElh,Sulh,nxlh,nylh,T):
   
   acum=np.zeros([nxlh,nylh])
   rh=np.zeros([nxlh+2,nylh+2])
   
   for i in range(1,nxlh+1):  
       for j in range(1,nylh+1):  
           acum[i-1,j-1]=Sulh[i-1,j-1]-(aPlh[i-1,j-1]*T[i,j]-aSlh[i-1,j-1]*T[i,j-1]-aNlh[i-1,j-1]*T[i,j+1]-aWlh[i-1,j-1]*T[i-1,j]-aElh[i-1,j-1]*T[i+1,j])
           rh[i,j]=acum[i-1,j-1]
                           
   residual = np.sqrt(np.mean(np.square(acum)))  #VRMS del valor residual

   return rh,residual

#________________________

def Parametros(nl,a):

    for i in range(1,nl):

        if a==0:        
            nxlh.append(int(nxlh[i-1]/2))
            nylh.append(int(nxlh[i-1]/2))                        
            dxlh.append((Lenght_x)/(nxlh[i]))
            dylh.append((Lenght_y)/(nylh[i]))

        if a==1:
            nxlh.append(int(nxlh[i-1]*2))
            nylh.append(int(nylh[i-1]*2))            
            dxlh.append((Lenght_x)/(nxlh[i]))
            dylh.append((Lenght_y)/(nxlh[i]))

            Tlh.append(np.zeros([nxlh[i]+2,nylh[i]+2]))

        Aew.append(dylh[i]*z)
        Asn.append(dxlh[i]*z)
        xlh.append(np.zeros(nxlh[i]+1))
        xclh.append(np.zeros(nxlh[i]+2))        
        ylh.append(np.zeros(nylh[i]+1))
        yclh.append(np.zeros(nylh[i]+2))        

        xlh[i],xclh[i]=Mallado(xclh[i], xlh[i], x0, lx, nxlh[i])
        ylh[i],xclh[i]=Mallado(yclh[i], ylh[i], y0, ly, nylh[i])
   
        aPlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
        aSlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
        aNlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
        aWlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
        aElh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))    
        Sulh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
   
        rlh.append(np.zeros([int(nxlh[i]+2),int(nylh[i]+2)]))
        elh.append(np.zeros([int(nxlh[i]+2),int(nylh[i]+2)]))        
#_______________________
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

#________________________
def Restriction(n_r):
    
    for i in range (1,n_r):    

        r_aux = np.zeros([nxlh[i],nylh[i]])
               
     #Restricción              
        rlh[i]=Prom(rlh[i-1],nxlh[i],nylh[i])      
       
        aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i]=Ecdiscreta(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],dxlh[i],dylh[i],z,Aew[i],Asn[i],kT,Tos,Ton,flux,Toe,nxlh[i],nylh[i]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

        for j in range(1,nxlh[i]+1):
            for k in range(1,nylh[i]+1):
                r_aux[j-1,k-1]=rlh[i][j,k]

        elh[i]=SOR(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],r_aux,nxlh[i],nylh[i],elh[i],10)      
    
        if i!=nl-1:
            for j in range(1,nylh[i]+1):
                for k in range(1,nxlh[i]+1):
                    rlh[i][j][k] = rlh[i][j][k]-(aPlh[i][j-1][k-1]*elh[i][j][k]-aSlh[i][j-1][k-1]*elh[i][j][k-1]-aNlh[i][j-1][k-1]*elh[i][j][k+1]-aWlh[i][j-1][k-1]*elh[i][j-1][k]-aElh[i][j-1][k-1]*elh[i][j+1][k])

#____________________________________________________________________

def Restriction_FMG(n_r):

    for i in range (1,n_r):    

        r_aux = np.zeros([nxlh[i-1],nylh[i-1]])
               
     #Restricción              
        rlh[i-1]=Prom(rlh[i],nxlh[i-1],nylh[i-1])      
       
#        aPlh[i-1],aSlh[i-1],aNlh[i-1],aWlh[i-1],aElh[i-1],Sulh[i-1]=Ecdiscreta(aPlh[i-1],aSlh[i-1],aNlh[i-1],aWlh[i-1],aElh[i-1],Sulh[i-1],dxlh[i-1],dylh[i-1],z,Aew[i-1],Asn[i-1],kT,Tos,Ton,flux,Toe,nxlh[i-1],nylh[i-1]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

        for j in range(1,nxlh[i-1]+1):
            for k in range(1,nylh[i-1]+1):
                r_aux[j-1,k-1]=rlh[i][j,k]

        elh[i-1]=SOR(aPlh[i-1],aSlh[i-1],aNlh[i-1],aWlh[i-1],aElh[i-1],r_aux,nxlh[i-1],nylh[i-1],elh[i-1],10)      
    
        if i!=nl-1:
            for j in range(1,nylh[i-1]+1):
                for k in range(1,nxlh[i-1]+1):
                    rlh[i-1][j][k] = rlh[i-1][j][k]-(aPlh[i-1][j-1][k-1]*elh[i-1][j][k]-aSlh[i-1][j-1][k-1]*elh[i-1][j][k-1]-aNlh[i-1][j-1][k-1]*elh[i-1][j][k+1]-aWlh[i-1][j-1][k-1]*elh[i-1][j-1][k]-aElh[i-1][j-1][k-1]*elh[i-1][j+1][k])
#____________________________________________________________________

def Prolongation(n_p):

    for i in range(1,n_p):

        elhp= np.zeros([nxlh[n_p-i-1]+2,nylh[n_p-i-1]+2])
        r_aux = np.zeros([nxlh[n_p-i-1],nylh[n_p-i-1]])

        elhp=Interpolation(elh[n_p-i],nxlh[n_p-i],nxlh[n_p-i-1],nylh[n_p-i],nylh[n_p-i-1])  
        elh[n_p-i-1]=elh[n_p-i-1]+elhp
    
        for j in range(1,nxlh[n_p-i-1]+1):
            for k in range(1,nylh[n_p-i-1]+1):
                r_aux[j-1,k-1]=rlh[n_p-i-1][j,k]

        elh[n_p-i-1]=SOR(aPlh[n_p-i-1],aSlh[n_p-i-1],aNlh[n_p-i-1],aWlh[n_p-i-1],aElh[n_p-i-1],r_aux,nxlh[n_p-i-1],nylh[n_p-i-1],elh[n_p-i-1],2)      

#____________________________________________________________________    

def Prolongation_FMG(n_p):

    for i in range(1,n_p):

        elhp= np.zeros([nxlh[i]+2,nylh[i]+2])
        r_aux = np.zeros([nxlh[i],nylh[i]])

        elhp=Interpolation(elh[i-1],nxlh[i-1],nxlh[i],nylh[i-1],nylh[i])  
        elh[i]=elh[i]+elhp
    
        for j in range(1,nxlh[i]+1):
            for k in range(1,nylh[i]+1):
                r_aux[j-1,k-1]=rlh[i][j,k]

        elh[i]=SOR(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],r_aux,nxlh[i],nylh[i],elh[i],2)      

#________________________

def Prom(rlh,nxlh,nylh):
   
    r2lh=np.zeros([nxlh+2,nylh+2])

    for i in range(1,nxlh):
        for j in range(1,nylh):  
            
            temp=rlh[2*i-1,2*j]+rlh[2*i+1,2*j]+rlh[2*i,2*j-1]+rlh[2*i,2*j+1]
            temp=2*temp+4*rlh[2*i,2*j]
            temp=temp+rlh[2*i-1,2*j-1]+rlh[2*i-1,2*j+1]+rlh[2*i+1,2*j-1]+rlh[2*i+1,2*j+1]
            temp=temp/16
            r2lh[i,j]=temp

    for i in range(1,nylh+1):  
            r2lh[i,nylh]=rlh[2*i-1,2*nylh-2]

    for j in range(1,nxlh+1):
            r2lh[nxlh,j]=rlh[2*nxlh-2,2*j-1]

    r2lh[nxlh,nylh]=rlh[2*nxlh-2,2*nxlh-2]
     
    return r2lh

#____________________________________________________________________
def Interpolation(e2lh,ei2h,eih,ej2h,ejh):    
 
    ehp=np.zeros([eih+2,ejh+2])  
   
    for i in range(1,ei2h+1):      
        for j in range(1,ej2h+1):
            ehp[2*i,2*j-1]=0.75*e2lh[i,j]+0.25*e2lh[i,j-1]
            ehp[2*i,2*j]=0.75*e2lh[i,j]+0.25*e2lh[i,j+1]

    for j in range(1,ejh+1):      
        for i in range(2,ei2h+1):
            ehp[2*i-1,j]=0.75*ehp[2*i,j]+0.25*ehp[2*i-2,j]

    for j in range(1,ejh+1):      
            ehp[1,j]=0.75*ehp[2,j]
            ehp[eih,j]=0.75*ehp[eih,j]
         
    return ehp

#________________________
def V_Cycle(nl):    
 
    Restriction(nl)
    Prolongation(nl)

#________________________
def W_Cycle(nl):    
 
    Restriction(nl)
    Prolongation(1)
    Restriction(1)
    Prolongation(2)
    Restriction(2)
    Prolongation(1)
    Restriction(1)
    Prolongation(nl)
#________________________
def F_Cycle(nl):

    Restriction(nl)
    Prolongation(1)
    Restriction(1)
    Prolongation(2)
    Restriction(2)
    Prolongation(nl)
#________________________

def FMG_Cycle(nl):
    
    for i in range(1,nl):                

        Tlh[i]=Interpolation(Tlh[i-1],nxlh[i-1],nxlh[i],nylh[i-1],nylh[i])  

        aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i]=Ecdiscreta(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],dxlh[i],dylh[i],z,Aew[i],Asn[i],kT,Tos,Ton,flux,Toe,nxlh[i],nylh[i]) #Se calculan los coeficientes aW,aE,ap,Su,Sp
        Tlh[i]=SOR(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],nxlh[i],nylh[i],Tlh[i],2)  
        rlh[i],residual=Residual(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],nxlh[i],nylh[i],Tlh[i])
    
        Restriction_FMG(i+1)
        Prolongation_FMG(i+1)

        Tlh[i]=Tlh[i]+elh[i]
        Tlh[i]=SOR(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],nxlh[i],nylh[i],Tlh[i],2)  
        rlh[i],residual=Residual(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],nxlh[i],nylh[i],Tlh[i])

"""
    Tlh[nl-1]=Tlh[nl-1]+elh[nl-1]
    Tlh[nl-1]=SOR(aPlh[nl-1],aWlh[nl-1],aElh[nl-1],Sulh[nl-1],nlh[nl-1],Tlh[nl-1],2)  
    rlh[nl-1],residual = Residual(aPlh[nl-1],aWlh[nl-1],aElh[nl-1],Sulh[nl-1],rlh[nl-1],nlh[nl-1],Tlh[nl-1])
""" 
#___________________________________________________________________

def boundaries(T,nx,ny,dx,dy,flux,k):

    T[:,ny+1]=100                   #frontera norte    
    T[0,:]=T[1,:]+flux*dx/(2*k)     #frontera oeste
    T[:,0]=T[:,1]                   #frontera sur
    T[nx+1,:]=T[nx,:]             #frontera este
   
    return T

#________________________
def Plot_T(T,x,y,dx,dy,lx,ly,nx,ny):
   
    T=np.transpose(T)
    X,Y=np.meshgrid(x,y)            #Definicion de la malla
    plt.contourf(X,Y,T, alpha=.75, cmap=plt.cm.hot)
    plt.colorbar()
    C = plt.contour(X, Y, T, colors='black', linewidth=.1)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Distribucion de Temperatura')
    plt.show()
#____________________________________________________________________

def Resverse():

    Tlh.reverse()

    nxlh.reverse()
    nylh.reverse()
    dxlh.reverse()
    dylh.reverse()
    
    xlh.reverse()
    xclh.reverse()
    ylh.reverse()
    yclh.reverse()
        
    aPlh.reverse()
    aWlh.reverse()
    aElh.reverse()
    aSlh.reverse()
    aNlh.reverse()
    Sulh.reverse()

    rlh.reverse()
    elh.reverse()
#____________________________________________________________________

def R_norm(n,r_0,residual):
    
    VRes2.append(residual/r_0)
    n=n+1
    Vnit2.append(n)
    t_par= time.process_time()
    end2=t_par-start
    VTim2.append(end2)

    return (n)

#____________________________________________________________________

def Multimalla_VWF(T,residual):

    n=0; iter_MM=0         
    Parametros(nl,0)    

    while iter_MM < max_iter and residual > tolerance:  

        T=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T,5)  
        rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T)

        if n==0:
            r_0=residual
            n=R_norm(n,r_0,residual)
    
        V_Cycle(nl)
#        W_Cycle(nl)
#        F_Cycle(nl)

        T=T+elh[0]
        rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T)
        T=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T,2)        
        rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T)

        n=R_norm(n,r_0,residual)

        iter_MM=iter_MM+1     

        for i in range (nl):
            elh[i]=np.zeros([int(nxlh[i]+2),int(nylh[i]+2)])

    T=boundaries(T,nxlh[0],nylh[0],dxlh[0],dylh[0],flux,kT)              #Se establecen las temperaturas de las fronteras

    return(T)
#____________________________________________________________________

def Multimalla_FMG(T,residual):

    n=0; iter_MM=0
    Parametros(nl,1)    

    Tlh[0]=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0],10)  
    rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0])

    r_0=residual
    n=R_norm(n,r_0,residual)

    FMG_Cycle(nl)
    rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0])

    Resverse()
    
    n=R_norm(n,r_0,residual)

    while iter_MM < max_iter and residual > tolerance:  

        Tlh[0]=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0],5)        
        rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0])
        
        Restriction(nl)
        Prolongation(nl)

        Tlh[0]=Tlh[0]+elh[0]
        rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0])
        Tlh[0]=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0],2)        
        rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],Tlh[0])

        n = R_norm(n,r_0,residual)
    
        iter_MM = iter_MM+1     

        for i in range (nl):
            elh[i]=np.zeros([int(nxlh[i]+2),int(nylh[i]+2)])

    Tlh[0]=boundaries(Tlh[0],nxlh[0],nylh[0],dxlh[0],dylh[0],flux,kT)              #Se establecen las temperaturas de las fronteras

    return(Tlh[0])       
#________________________
#Programa principal

xlh[0],xclh[0]=Mallado(xclh[0], xlh[0], x0, lx, nxlh[0])
ylh[0],yclh[0]=Mallado(yclh[0], ylh[0], y0, ly, nylh[0])
aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0]=Ecdiscreta(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],dxlh[0],dylh[0],z,Aew[0],Asn[0],kT,Tos,Ton,flux,Toe,nxlh[0],nylh[0]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

#T=Multimalla_VWF(T,residual)
#plt.figure(1)
#Plot_T(T,xclh[0],yclh[0],dxlh[0],dylh[0],lx,ly,nxlh[0],nylh[0])                 #Se grafican los resultados

Tlh[0]=Multimalla_FMG(Tlh[0],residual)
plt.figure(1)
Plot_T(Tlh[0],xclh[0],yclh[0],dxlh[0],dylh[0],lx,ly,nxlh[0],nylh[0])                 #Se grafican los resultados

#plt.figure(2)
#plt.subplot(221)
#plt.plot(nit,Res)
#plt.subplot(222)
#plt.plot(nit,Tim)
#plt.subplot(223)
#plt.plot(Tim,Res)

#print(T)
#print(iter_MM)
#print(residual)
end= time.process_time()
print(end-start)



