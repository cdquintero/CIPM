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
x0=0.0; lx=1
y0=0.0; ly=1                                       #Longitu
z=0.01

Lenght_x = lx-x0
Lenght_y = ly-y0

nx=16; ny=16
dx=(lx-x0)/(nx); dy=(ly-y0)/(ny)

x=np.zeros(nx+1); y=np.zeros(ny+1)
xc=np.zeros(nx+2); yc=np.zeros(ny+2)

Tos=0; Ton=100; Toe=0; flux=500000;                          #Condiciones de Frontera
k=1000; Aew=dy*z; Asn=dx*z                                   #Variables del Problema

ap=np.zeros([nx,ny]); aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny]); Su=np.zeros([nx,ny])

T=np.zeros([nx+2,ny+2]); 

A=np.zeros(ny+2); Cp=np.zeros(ny+2); C=np.zeros(ny+2)

Res=[]
Tim=[]
nit=[]

#Ecuaciones discretizadas: Esta funcion calcula los coeficientes aps
#____________________________________________________________________

def Ecdiscreta(dx,dy,z,Aew,Asn,k,Tos,Ton,flux,Toe,nx,ny):
    
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

            ap[i,j]=aW[i,j]+aE[i,j]+aS[i,j]+aN[i,j]-Sp
       
    return  aW, ap, aE, aS, aN, Su

#Solucionador
#____________________________________________________________________

def Sor_2D(param,aS,ap,aN,aW,aE,Su,nx,ny,f):
    
    tolerance=1e-7; max_iter=100000; count_iter=0; residual=1.0        #Criterios de Convergencia
    m=0; r_0=0 

    f0=np.zeros([nx+2,ny+2])
    acum=np.zeros([nx,ny])

    while count_iter <= max_iter and residual >  tolerance:  
    
        for i in range(nx+2):
                for jj in range(ny+2):
                    f0[i,jj]=f[i,jj]

        for i in range(1,nx+1):
            for j in range(1,ny+1):           
                    f[i,j]=(1.0-param)*f0[i,j]+param*(Su[i-1,j-1]+aW[i-1,j-1]*f[i-1,j]+aE[i-1,j-1]*f[i+1,j]+aS[i-1,j-1]*f[i,j-1]+aN[i-1,j-1]*f[i,j+1])/ap[i-1,j-1]
   
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo
        
        for i in range(1,nx+1):  
            for j in range(1,ny+1):  
                acum[i-1,j-1]=Su[i-1,j-1]-(ap[i-1,j-1]*T[i,j]-aS[i-1,j-1]*T[i,j-1]-aN[i-1,j-1]*T[i,j+1]-aW[i-1,j-1]*T[i-1,j]-aE[i-1,j-1]*T[i+1,j])
                
        residual = np.sqrt(np.mean(np.square(acum)))  #VRMS del valor residual

        if m==0:
            r_0=residual

        Res.append(residual/r_0)
        m=m+1
        nit.append(m)   
        t_par= time.process_time()
        end2=t_par-start
        Tim.append(end2)
    
    return f,count_iter,residual
#____________________________________________________________________

def GaussSeidel(aS,ap,aN,aW,aE,Su,nx,ny,f):

    tolerance=1e-7; max_iter=100000; count_iter=0; residual=1.0        #Criterios de Convergencia
    m=0; r_0=0

    acum=np.zeros([nx,ny])

    while count_iter <= max_iter and residual > tolerance:  
      
        for i in range(1,nx+1):
            for j in range(1,ny+1):
                 f[i,j]=(Su[i-1,j-1]+aW[i-1,j-1]*f[i-1,j]+aE[i-1,j-1]*f[i+1,j]+aS[i-1,j-1]*f[i,j-1]+aN[i-1,j-1]*f[i,j+1])/ap[i-1,j-1]
      
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo
   
        for i in range(1,nx+1):  
            for j in range(1,ny+1):  
                acum[i-1,j-1]=Su[i-1,j-1]-(ap[i-1,j-1]*T[i,j]-aS[i-1,j-1]*T[i,j-1]-aN[i-1,j-1]*T[i,j+1]-aW[i-1,j-1]*T[i-1,j]-aE[i-1,j-1]*T[i+1,j])
                
        residual = np.sqrt(np.mean(np.square(acum)))  #VRMS del valor residual

        if m==0:
            r_0=residual

        Res.append(residual/r_0)
        m=m+1
        nit.append(m)   
        t_par= time.process_time()
        end2=t_par-start
        Tim.append(end2)
          
    return f,count_iter,residual

#Condiciones de frontera: Esta funcion establece los valores de temperatura de las fronteras
#____________________________________________________________________

def Thomas(aS,ap,aN,aW,aE,Su,nx,ny,Cp,A,C,f):

    tolerance=1e-7; max_iter=100000; count_iter=0; residual=1.0        #Criterios de Convergencia
    m=0; r_0=0

    acum=np.zeros([nx,ny])

    while count_iter <= max_iter and residual > tolerance:  
    
        Cp[0]=Tos
        #Cp[ny-1]=f[i,ny-1] 
                    
        for i in range(1,nx+1):
            for j in range(1,ny+1):
                    C[j]=aW[i-1,j-1]*f[i-1,j]+aE[i-1,j-1]*f[i+1,j]+Su[i-1,j-1]
                    A[j]=aN[i-1,j-1]/(ap[i-1,j-1]-(aS[i-1,j-1]*A[j-1]))
                    Cp[j]=(aS[i-1,j-1]*Cp[j-1]+C[j])/(ap[i-1,j-1]-aS[i-1,j-1]*A[j-1])
                        
            f[i,ny+1]=A[ny+1]*Ton+Cp[ny+1]
                        
            for k in range(ny,-1,-1):
                f[i,k]=A[k]*f[i,k+1]+Cp[k]
            
        count_iter=count_iter+1                                       #Numero de iteraciones del codigo

        for i in range(1,nx+1):  
            for j in range(1,ny+1):  
                acum[i-1,j-1]=Su[i-1,j-1]-(ap[i-1,j-1]*T[i,j]-aS[i-1,j-1]*T[i,j-1]-aN[i-1,j-1]*T[i,j+1]-aW[i-1,j-1]*T[i-1,j]-aE[i-1,j-1]*T[i+1,j])
                
        residual = np.sqrt(np.mean(np.square(acum)))  #VRMS del valor residual

        if m==0:
            r_0=residual

        Res.append(residual/r_0)
        m=m+1
        nit.append(m)   
        t_par= time.process_time()
        end2=t_par-start
        Tim.append(end2)
              
    return f,count_iter,residual
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
#____________________________________________________________________

def boundaries(T,nx,ny,dx,dy,flux,k):

    T[:,ny+1]=100                   #frontera norte    
    T[0,:]=T[1,:]+flux*dx/(2*k)     #frontera oeste
    T[:,0]=T[:,1]                   #frontera sur
    T[nx+1,:]=T[nx,:]             #frontera este
   
    return T

#Grafica
#____________________________________________________________________

def Plot_T(T,x,y,dx,dy,lx,ly,nx,ny):

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

x,xc = Mallado(xc, x, x0, lx, nx)
y,yc = Mallado(yc, y, y0, ly, ny)
aW, ap, aE, aS, aN, Su = Ecdiscreta(dx,dy,z,Aew,Asn,k,Tos,Ton,flux,Toe,nx,ny) #Se calculan los coeficientes aW,aE,ap,Su,Sp

#T,ci,r =Sor_2D(1.2,aS,ap,aN,aW,aE,Su,nx,ny,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de SOR
#T,ci,r =GaussSeidel(aS,ap,aN,aW,aE,Su,nx,ny,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de SOR
T,ci,r=Thomas(aS,ap,aN,aW,aE,Su,nx,ny,Cp,A,C,T)   #Se resuelve la matriz Tridiagonal por el algoritmo de Thomas


T=boundaries(T,nx,ny,dx,dy,flux,k)              #Se establecen las temperaturas de las fronteras
 
plt.figure(1)
Plot_T(T,xc,yc,dx,dy,lx,ly,nx,ny)                 #Se grafican los resultados

plt.figure(2)
plt.subplot(221)
plt.plot(nit,Res)
plt.subplot(222)
plt.plot(nit,Tim)
plt.subplot(223)
plt.plot(Tim,Res)



#print(T)
#print(ci)
#print(r)
end= time.process_time()
print(end-start)
