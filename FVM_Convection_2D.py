# -- coding: utf-8 --
"""
Spyder Editor

This is a temporary script file.
"""
import time
start= time.process_time()

import numpy as np
import matplotlib.pyplot as plt
import math

#Definicion de variables
lx=1.0; ly=1.0; z=0.01                                       #Longitud
dx=0.0625; dy=0.0625
nx=math.ceil(lx/dx); ny=math.ceil(ly/dy); n=nx*ny        #Numero de nodos
x=np.zeros(nx+2); y=np.zeros(ny+2)                               #Malla Graficas

QB=0; QA=1; 

u=0.1; rho=1; Gamma=0.1
D=Gamma/dx; F=rho*u

Tow=0; Ton=20; Toe=0;                           #Condiciones de Frontera
#k=1000; Aew=dy*z; Asn=dx*z                                   #Variables del Problema

aP=np.zeros([nx,ny]); aW=(D+F/2)*np.ones([nx,ny]); aE=(D-F/2)*np.ones([nx,ny])
aS=(D+F/2)*np.ones([nx,ny]); aN=(D-F/2)*np.ones([nx,ny]); 


Su=np.zeros([nx,ny])
T=np.zeros([nx+2,ny+2]); acum=np.zeros([nx,ny])
A=np.zeros(ny); Cp=np.zeros(ny); C=np.zeros(ny)

#Ecuaciones discretizadas: Esta funcion calcula los coeficientes aps
#________________________
def Ecdiscreta(dx,dy,z,D,F,u,QA,QB,nx,ny):
         
    bW=(2*D+F)*QB                #Fuentes de calor en las fronteras
    bE=(2*D-F)*QB 
    bS=(2*D+F)*QA 
    bN=(2*D-F)*QB 

    for i in range(nx):             #i+1 columna del nodo
        for j in range(ny):         #j+1 fila de nodo

            Sp=0                        #Nodos Centrales          
            Su[i,j]=0            
           
            if j==0:                    #Frontera sur
                aS[i,j]=0
                Sp=-(2*D+F)
                Su[i,j]=bS                
               
            if j==ny-1:                 #Frontera norte
                aN[i,j]=0
                Sp=-(2*D-F)
                Su[i,j]=bN
           
            if i==0:               #Frontera oeste
                aW[i,j]=0

                if j==0:                #Borde inferior izquierdo
                    Sp=-2*(2*D+F)
                    Su[i,j]=bW+bS
                
                elif j==ny-1:           #Borde superior izquierdo
                    Sp=-2*(2*D+F)
                    Su[i,j]=bW+bN
             
                else:                   #Borde izquierdo
                    Sp=-(2*D+F)    
                    Su[i,j]=bW
                           
            elif i==nx-1:         #Frontera este
                aE[i,j]=0
               
                if j==0:                #Borde inferior derecho
                    Sp=-2*(2*D-F)   
                    Su[i,j]=bE+bS
               
                elif j==ny-1:           #Borde superior derecho
                    Sp=-2*(2*D-F)
                    Su[i,j]=bE+bN
               
                else:                   #Borde derecho
                    Sp=-(2*D-F)
                    Su[i,j]=bE

            aP[i,j]=aW[i,j]+aE[i,j]+aS[i,j]+aN[i,j]-Sp
       
    return  aW, aP, aE, aS, aN, Su

#Solucionador
#________________________

def Sor_2D(param,aS,aP,aN,aW,aE,Su,nx,ny,Phi):

    tolerance=1e-7; max_iter=100000; count_iter=0; residual=1.0        #Criterios de Convergencia

    while count_iter <= max_iter and residual >  tolerance:  
   
        Phi0=Phi                                                         #Valor inicial de la funcion
   
        for i in range(1,nx-1):
            for j in range(1,ny-1):
           
                    Phi[i,j]=(1.0-param)*Phi0[i,j]+param*(Su[i-1,j-1]+aW[i-1,j-1]*Phi[i-1,j]+aE[i-1,j-1]*Phi[i+1,j]+aS[i-1,j-1]*Phi[i,j-1]+aN[i-1,j-1]*Phi[i,j+1])/aP[i-1,j-1]
   
        for i in range(1,nx-1):  
            for j in range(1,ny-1):  
                acum[i-1,j-1]=Su[i-1,j-1]+aW[i-1,j-1]*T[i-1,j]+aE[i-1,j-1]*T[i+1,j]+aS[i-1,j-1]*T[i,j-1]+aN[i-1,j-1]*T[i,j+1]-aP[i-1,j-1]*T[i,j]                
               
        residual = np.sqrt(np.mean(np.square(acum)))  #VRMS del valor residual
   
    return Phi,residual
#________________________

def boundaries(T,nx,ny,dx,dy,QA,QB):
    T[:,0]=QA     #frontera sur
    T[:,ny-1]=QB                    #frontera norte
    T[0,:]=QB                  #frontera este
    T[nx-1,:]=QB            #frontera oeste
   
    return T

#Grafica
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

aW, aP, aE, aS, aN, Su=Ecdiscreta(dx,dy,z,D,F,u,QA,QB,nx,ny) #Se calculan los coeficientes aW,aE,ap,Su,Sp

T,r =Sor_2D(1.2,aS,aP,aN,aW,aE,Su,nx+2,ny+2,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de SOR
#T,ci,r =GaussSeidel(aS,aP,aN,aW,aE,Su,nx,ny,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de SOR
#T,ci,r=Thomas(aS,aP,aN,aW,aE,Su,n,nx,ny,Cp,A,C,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de Thomas


T=boundaries(T,nx+2,ny+2,dx,dy,QA,QB)              #Se establecen las temperaturas de las fronteras
 
Plot_T(T,x,y,dx,dy,lx,ly,nx+2,ny+2)                 #Se grafican los resultados
print(T)
print(r)
end= time.process_time()
print(end-start)