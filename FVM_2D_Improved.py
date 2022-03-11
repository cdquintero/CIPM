# -*- coding: utf-8 -*-
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
lx=0.3; ly=0.4; z=0.01                                       #Longitud 
dx=0.05; dy=0.05                                             #Pasos
nx=math.ceil(lx/dx)+2; ny=math.ceil(ly/dy)+2; n=nx*ny        #Numero de nodos
x=np.zeros(nx); y=np.zeros(ny)                               #Malla Graficas
Tos=0; Ton=100; Toe=0; flux=500000;                          #Condiciones de Frontera
k=1000; Aew=dy*z; Asn=dx*z                                   #Variables del Problema

ap=np.zeros([nx,ny]); aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny]); Su=np.zeros([nx,ny])
T=np.zeros([nx,ny]); acum=np.zeros([nx,ny])
A=np.zeros(ny); Cp=np.zeros(ny); C=np.zeros(ny)

#Ecuaciones discretizadas: Esta funcion calcula los coeficientes aps
#____________________________________________________________________

def Ecdiscreta(dx,dy,z,Aew,Asn,k,Tos,Ton,flux,Toe,nx,ny):
    
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

            ap[i,j]=aW[i,j]+aE[i,j]+aS[i,j]+aN[i,j]-Sp
       
    return  aW, ap, aE, aS, aN, Su

#Solucionador
#____________________________________________________________________

def Sor_2D(param,aS,ap,aN,aW,aE,Su,nx,ny,f):

    tolerance=1e-7; max_iter=1000; count_iter=0; residual=1.0        #Criterios de Convergencia

    while count_iter <= max_iter and residual > tolerance:  
    
        f0=f                                                         #Valor inicial de la funcion
   
        for i in range(1,nx-1):
            for j in range(1,ny-1):
           
                    f[i,j]=(1.0-param)*f0[i,j]+param*(Su[i,j]+aW[i,j]*f[i-1,j]+aE[i,j]*f[i+1,j]+aS[i,j]*f[i,j-1]+aN[i,j]*f[i,j+1])/ap[i,j]
   
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo
        for i in range(1,nx-1):  
            for j in range(1,ny-1):  
                 
                residual = np.sqrt(np.mean(np.square(f[i,j]-acum[i,j])))  #VRMS del valor residual
                acum[i,j] = f[i,j]

    return f,count_iter,residual
#____________________________________________________________________

def GaussSeidel(aS,ap,aN,aW,aE,Su,nx,ny,f):

    tolerance=1e-7; max_iter=10000; count_iter=0; residual=1.0        #Criterios de Convergencia

    while count_iter <= max_iter and residual > tolerance:  
      
        for i in range(1,nx-1):
            for j in range(1,ny-1):
           
                f[i,j]=(Su[i,j]+aW[i,j]*f[i-1,j]+aE[i,j]*f[i+1,j]+aS[i,j]*f[i,j-1]+aN[i,j]*f[i,j+1])/ap[i,j]
      
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo
        for i in range(1,nx-1):  
            for j in range(1,ny-1):  
                 
                residual = np.sqrt(np.mean(np.square(f[i,j]-acum[i,j])))  #VRMS del valor residual
                acum[i,j] = f[i,j]

    return f,count_iter,residual

#Condiciones de frontera: Esta funcion establece los valores de temperatura de las fronteras
#____________________________________________________________________

def Thomas(aS,ap,aN,aW,aE,Su,n,nx,ny,Cp,A,C,f):

    tolerance=1e-7; max_iter=10000; count_iter=0; residual=1.0        #Criterios de Convergencia

    while count_iter <= max_iter and residual > tolerance:  
    
        Cp[0]=Tos
        #Cp[ny-1]=f[i,ny-1] 
                    
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                    C[j]=aW[i,j]*f[i-1,j]+aE[i,j]*f[i+1,j]+Su[i,j]
                    A[j]=aN[i,j]/(ap[i,j]-(aS[i,j]*A[j-1]))
                    Cp[j]=(aS[i,j]*Cp[j-1]+C[j])/(ap[i,j]-aS[i,j]*A[j-1])
                        
            f[i,ny-1]=A[ny-1]*Ton+Cp[ny-1]
                        
            for k in range(ny-2,-1,-1):
                f[i,k]=A[k]*f[i,k+1]+Cp[k]
            
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo
        for i in range(1,nx-1):  
            for j in range(1,ny-1):  
                 
                residual = np.sqrt(np.mean(np.square(f[i,j]-acum[i,j])))  #VRMS del valor residual
                acum[i,j] = f[i,j]
              
    return f,count_iter,residual
#____________________________________________________________________

def boundaries(T,nx,ny,dx,dy,flux,k):
    T[:,0]=T[:,1]                   #frontera oeste
    T[:,ny-1]=100                   #frontera este
    T[0,:]=T[1,:]+flux*dx/(2*k)     #frontera sur
    T[nx-1,:]=T[nx-2,:]             #frontera norte
    
    return T

#Grafica
#____________________________________________________________________

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

#____________________________________________________________________
#Programa principal

aW, ap, aE, aS, aN, Su=Ecdiscreta(dx,dy,z,Aew,Asn,k,Tos,Ton,flux,Toe,nx,ny) #Se calculan los coeficientes aW,aE,ap,Su,Sp

T,ci,r =Sor_2D(1.2,aS,ap,aN,aW,aE,Su,nx,ny,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de SOR
#T,ci,r =GaussSeidel(aS,ap,aN,aW,aE,Su,nx,ny,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de SOR
#T,ci,r=Thomas(aS,ap,aN,aW,aE,Su,n,nx,ny,Cp,A,C,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de Thomas


T=boundaries(T,nx,ny,dx,dy,flux,k)              #Se establecen las temperaturas de las fronteras
 
Plot_T(T,x,y,dx,dy,lx,ly,nx,ny)                 #Se grafican los resultados
print(T)
print(ci)
print(r)
end= time.process_time()
print(end-start)
