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
nl=3

lx=1; ly=1; z=0.01
dxlh=[0.0125]; dylh=[0.0125]
nxlh=[math.ceil(lx/dxlh[0]+2)]; nylh=[math.ceil(ly/dylh[0]+2)]
x=np.zeros(int(nxlh[0])); y=np.zeros(int(nylh[0]))
aWlh=[np.zeros([int(nxlh[0]),int(nylh[0])])]; aElh=[np.zeros([int(nxlh[0]),int(nylh[0])])]
aSlh=[np.zeros([int(nxlh[0]),int(nylh[0])])]; aNlh=[np.zeros([int(nxlh[0]),int(nylh[0])])]
aPlh=[np.zeros([int(nxlh[0]),int(nylh[0])])]; Sulh=[np.zeros([int(nxlh[0]),int(nylh[0])])]

rlh=[np.zeros([int(nxlh[0]),int(nylh[0])])];
elh=[np.zeros([int(nxlh[0]),int(nylh[0])])];
elhp=[0]
N=0      

T=np.zeros([int(nxlh[0]),int(nylh[0])]); acum=np.zeros([int(nxlh[0]),int(nylh[0])])
rh=np.zeros([int(nxlh[0]),int(nylh[0])])

Itrpx=[0];Itrpy=[0]
Rstrx=[0];Rstry=[0]
Tos=0; Ton=100; Toe=0; flux=500000;                          #Condiciones de Frontera
kT=1000; Aew=dylh[0]*z; Asn=dxlh[0]*z

tolerance=1e-7; max_iter=10000; count_iter=0; residual=1.0; iter_GS=0

#Ecuaciones discretizadas
#________________________
def Ecdiscreta(aP,aS,aN,aW,aE,Su,dx,dy,z,Aew,Asn,k,Tos,Ton,flux,Toe,nx,ny):
 
    for i in range(1,nx-1):             #i+1 columna del nodo
        for j in range(1,ny-1):         #j+1 fila de nodo

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
         
            if j==1:                    #Frontera sur
                aS[i,j]=0
                Su[i,j]=2*k*Asn*Tos/dy
               
            if j==ny-2:                 #Frontera norte
                aN[i,j]=0
                Sp=-2*k*Asn/dy
                Su[i,j]=2*k*Asn*Ton/dy
           
            if i==1:               #Frontera oeste
              aW[i,j]=0

              if j==1:                #Borde inferior izquierdo
                  Su[i,j]=bW+bS
               
              elif j==ny-2:           #Borde superior izquierdo
                  Sp=-2*k*Asn/dy
                  Su[i,j]=bW+bN
             
              else:                   #Borde izquierdo
                    Su[i,j]=bW
                           
            elif i==nx-2:         #Frontera este
                aE[i,j]=0
                               
                if j==1:                #Borde inferior derecho
                    Su[i,j]=bE+bS
               
                elif j==ny-2:           #Borde superior derecho
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

    while count_iter <= max_iter and residual > tolerance:  
        f0=f                                                         #Valor inicial de la funcion
        for i in range(1,nx-1):
            for j in range(1,ny-1):        
                f[i,j]=(1.0-param)*f0[i,j]+param*(Su[i,j]+aW[i,j]*f[i-1,j]+aE[i,j]*f[i+1,j]+aS[i,j]*f[i,j-1]+aN[i,j]*f[i,j+1])/aP[i,j]
 
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo

    return f
#________________________
def Residual(aPlh,aSlh,aNlh,aWlh,aElh,Sulh,nxlh,nylh,T):
      
   rh=np.zeros([nxlh,nylh])
   
   for i in range(1,nxlh-1):  
       for j in range(1,nylh-1):  
           rh[i,j]=Sulh[i,j]-(aPlh[i,j]*T[i,j]-aSlh[i,j]*T[i,j-1]-aNlh[i,j]*T[i,j+1]-aWlh[i,j]*T[i-1,j]-aNlh[i,j]*T[i+1,j])
           acum[i,j]=Sulh[i,j]+aWlh[i,j]*T[i-1,j]+aElh[i,j]*T[i+1,j]+aSlh[i,j]*T[i,j-1]+aNlh[i,j]*T[i,j+1]-aPlh[i,j]*T[i,j]                
                
   residual = np.sqrt(np.mean(np.square(acum)))  #VRMS del valor residual

   return rh,residual

#________________________

def Parametros(i):

    dxlh.append((2**i)*dxlh[0])
    dylh.append((2**i)*dxlh[0])
    nxlh.append(math.ceil(lx/dxlh[i])+2)
    nylh.append(math.ceil(ly/dylh[i])+2)
   
    aPlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
    aSlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
    aNlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
    aWlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
    aElh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))    
    Sulh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
   
    rlh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
    elh.append(np.zeros([int(nxlh[i]),int(nylh[i])]))        
    elhp.append(np.zeros([int(nxlh[i]),int(nylh[i])]))
   
    Itrpx.append(np.zeros([int(nxlh[i-1]),int(nylh[i])]))
    Rstrx.append(np.zeros([int(nxlh[i]),int(nylh[i-1])]))
    Itrpy.append(np.zeros([int(nxlh[i-1]),int(nylh[i])]))
    Rstry.append(np.zeros([int(nxlh[i]),int(nylh[i-1])]))  
   
#_______________________

def Restriction(n_r):

    for i in range (1,n_r):    
    #Definicion de Parámetros    
        if iter_GS==0:
            Parametros(i)
                
     #Restricción              
        rlh[i],elh[i]=Prom(aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],Sulh[i],rlh[i-1],nxlh[i],nylh[i],dxlh[i],dylh[i],z,Aew,Asn,kT,Tos,Ton,flux,Toe)      
       
        if i!=nl-1:
            for k in range(1,nylh[i]-1):
                for j in range(1,nxlh[i]-1):
                    rlh[i][k][j]=rlh[i][k][j]+aWlh[i][k][j]*elh[i][k][j-1]+aElh[i][k][j]*elh[i][k][j+1]-aPlh[i][k][j]*elh[i][k][j]

#____________________________________________________________________

def Prolongation(n_p):

    for i in range(1,n_p):
       
        elhp[i]=Interpolation(elh[nl-i],nxlh[nl-i],nxlh[nl-i-1],nylh[nl-i],nylh[nl-i-1])  
        elh[nl-i-1]=elh[nl-i-1]+elhp[i]

    
#________________________

def Prom(aPlh,aSlh,aNlh,aWlh,aElh,Sulh,rlh,nxlh,nylh,dxlh,dylh,z,Aew,Asn,kT,Tos,Ton,flux,Toe):
    r2lh=np.zeros([nxlh,nylh])
    e2lh=np.zeros([nxlh,nylh])

    for i in range(1,nxlh-1): 
        for j in range(1,nylh-1):  
            temp=rlh[2*i-1,2*j]+rlh[2*i+1,2*j]+rlh[2*i,2*j-1]+rlh[2*i,2*j+1]
            temp=2*temp+4*rlh[2*i,2*j]
            temp=temp+rlh[2*i-1,2*j-1]+rlh[2*i-1,2*j+1]+rlh[2*i+1,2*j-1]+rlh[2*i+1,2*j+1]
            temp=temp/16
            r2lh[i,j]=temp 
    
    aPlh,aSlh,aNlh,aWlh,aElh,Sulh=Ecdiscreta(aPlh,aSlh,aNlh,aWlh,aElh,Sulh,dxlh,dylh,z,Aew,Asn,kT,Tos,Ton,flux,Toe,nxlh,nylh) #Se calculan los coeficientes aW,aE,ap,Su,Sp
    e2lh=SOR(aPlh,aSlh,aNlh,aWlh,aElh,r2lh,nxlh,nylh,e2lh,10)      
 
    return r2lh,e2lh

#____________________________________________________________________
def Interpolation(e2h,nx2h,nxh,ny2h,nyh):    
 
    ehp=np.zeros([nxh,nyh])
    
    for i in range(1,nx2h-1): 
        for j in range(1,ny2h-1):
            ehp[2*i,2*j]=e2h[i,j]
            
    for i in range(1,nx2h-1): 
        for j in range(1,ny2h-1): 
            ehp[2*i,2*j-1]=1/2*(e2h[i,j]+e2h[i,j-1])
            
    for i in range(1,nx2h-1): 
        for j in range(1,nyh-1): 
            ehp[2*i-1,j]=1/2*(ehp[2*i,j]+ehp[2*i-1,j])
             
#ehp[2*i+1,2*j]=1/2*(e2h[i,j]+e2h[i+1,j])
#ehp[2*i+1,2*j+1]=1/4*(e2h[i,j]+e2h[i+1,j]+e2h[i,j+1]+e2h[i+1,j+1])
         
    return ehp

#________________________


def M_Interpolation(nx,nx2,ny,ny2,Itrpx,Itrpy):

    Vx=np.ones(nx2)
    Itrpx[1,1]=Vx[1]
    Itrpx[nx-2,nx2-2]=2*Vx[nx2-1]
   
    for i in range(1,int(nx/2-1)):
        Itrpx[2*i,i]=2*Vx[1]
        Itrpx[2*i+1,i]=Vx[1]
        Itrpx[2*i+1,i+1]=Vx[1]        

    Vy=np.ones(ny2)
    Itrpy[1,1]=Vy[1]
    Itrpy[nx-2,ny2-2]=2*Vy[ny2-1]
   
    for i in range(1,int(ny/2-1)):
        Itrpy[2*i,i]=2*Vy[1]
        Itrpy[2*i+1,i]=Vy[1]
        Itrpy[2*i+1,i+1]=Vy[1]  
 
    return Itrpx,Itrpy    
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

def boundaries(T,nx,ny,dx,dy,flux,k):
    T[:,0]=T[:,1]                   #frontera oeste
    T[:,ny-1]=100                   #frontera este
    T[0,:]=T[1,:]+flux*dx/(2*k)     #frontera sur
    T[nx-1,:]=T[nx-2,:]             #frontera norte
   
    return T

#________________________
def Plot_T(T,x,y,dx,dy,lx,ly,nx,ny):
    x[1]=dx/2                       #Condiciones de medio paso para fronteras eje x
    x[nx-1]=lx
    y[1]=dy/2                       #Condiciones de medio paso para fronteras eje y
    y[ny-1]=ly
   
    for i in range(2,nx-1):  
        x[i]=x[i-1]+dx              #Definicion eje x
    for j in range(2,ny-1):  
        y[j]=y[j-1]+dy              #Definicion eje y
   
    T=np.transpose(T)
    X,Y=np.meshgrid(x,y)            #Definicion de la malla
    plt.contourf(X,Y,T, alpha=.75, cmap=plt.cm.hot)
    plt.colorbar()
    C = plt.contour(X, Y, T, colors='black', linewidth=.1)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Temperature Distribution')
    plt.show()        
#________________________
#Programa principal

aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0]=Ecdiscreta(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],dxlh[0],dylh[0],z,Aew,Asn,kT,Tos,Ton,flux,Toe,nxlh[0],nylh[0]) #Se calculan los coeficientes aW,aE,ap,Su,Sp

while iter_GS < max_iter and residual > tolerance:  
       
    T=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T,5)  
    rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T)

    V_Cycle(nl)

    T=T+elhp[nl-1]
    T=SOR(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T,2)        
    rlh[0],residual=Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Sulh[0],nxlh[0],nylh[0],T)
   
    iter_GS=iter_GS+1  
   
T=boundaries(T,nxlh[0],nylh[0],dxlh[0],dylh[0],flux,kT)              #Se establecen las temperaturas de las fronteras
 
Plot_T(T,x,y,dxlh[0],dylh[0],lx,ly,nxlh[0],nylh[0])                 #Se grafican los resultados
print(T)
#print(ci)
#print(r)
end= time.process_time()
print(end-start)

