
import time
start1= time.process_time()

import numpy as np
import matplotlib.pyplot as plt
import math

#___________________________________________________________________-
#DEFINICIÓN DE VARIABLES

x0=0.0; xl=3.0
y0=0.0; yl=1.0                                       #Longitud

nx=64; ny=64               

dx=(xl-x0)/(nx);    dy=(yl-y0)/(ny);    dV=dx*dy;   dt=2.0E-4

x=np.zeros(nx+2); y=np.zeros(ny+2)                               #Malla Graficas

Ae=dy;  Aw=dy
An=dx;  As=dx

Lambda=0.99

itmax=8000

tc=np.zeros(itmax)
Nu_g=np.zeros(itmax)

max_iter1=2000 #solver
max_iter2=500 #fixed
tolerance=5.0E-6
Ra=100
alpha=0*(np.pi/180.0)


n=1 #counter to define steady state
n_mean=22000 #number of norms infinite of T to be averaged

#convt=1.0 #Convergence of TT
residual=1
norm_inf=1.0 #norm infinite to define steady state

m=0
Tm=0
Fm=0
#___________________________________________________________________-
#DEFINICIÓN DE ARREGLOS

#Arreglos de Tamaño n
aP=np.zeros([nx,ny]); aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny]);  
Su=np.zeros([nx,ny]); Sp=np.zeros([nx,ny]);
div=np.zeros([nx,ny])

acum=np.zeros([nx,ny])
A=np.zeros(ny); Cp=np.zeros(ny); C=np.zeros(ny)
Nu= np.zeros(nx); steady_T= np.zeros(n_mean)

u= np.zeros([nx+1,ny+2]); v= np.zeros([nx+2,ny+1])
#Arreglos de Tamaño n+1
x= np.zeros([nx+1]); y=np.zeros([ny+1])
F=np.zeros([nx+1,ny+1])

#Arreglos de Tamaño n+2
uc= np.zeros([nx+2,ny+2]); vc= np.zeros([nx+2,ny+2])
xc= np.zeros([nx+2]); yc= np.zeros([ny+2])

convt= np.ones([nx+2,ny+2])
T=np.zeros([nx+2,ny+2])
TT=np.zeros([nx+2,ny+2])

T[:,0]=1.0
T[:,ny+1]=0.0

TT[:,0]=1.0
TT[:,ny+1]=0.0

Nu=np.zeros(nx+2)
Ng=0

Res=[]
TRes=[]
FRes=[]
Tim=[]
TTim=[]
FTim=[]
cTim=[]
nit=[]
Tnit=[]
Fnit=[]
dTemp=[]


#_________________________________________________________________________________

#________________________
# Nusselt Global on hot surface
def Nusselt(nx,dx,dy,T):
   
    Nu=np.zeros(nx+2)

    for i in range (1,nx+1):
        Nu[i]=-2.0*(T[i,nx+1]-T[i,nx])/dy
   
    Nu_g=0.0

    for i in range (1,nx+1):
        Nu_g= Nu_g + Nu[i]*dx
 
    return Nu,Nu_g
#________________________
#Divergence
def Divergence(nx,ny,u,v):

    Div=np.zeros([nx,ny])

    for i in range (1,nx):
        for j in range (1,ny):
            Div[i,j]=u[i,j]-u[i-1,j]+v[i,j]-v[i,j-1]

    Div_max = np.maximum(np.abs(Div))

    return Div,Div_max
#Solucionador
#________________________

def Sor_2D(param,Phi,nx,ny,aP,aE,aW,aN,aS,Sp,max_iter,tolerance,residual,m,a,Xm):              

    count_iter=0
    residual=1
    Phi0=np.zeros([nx,ny])
   
    while count_iter <= max_iter and residual > tolerance:  
#        Phi0=Phi                                                         #Valor inicial de la funcion
        for i in range(nx):
            for jj in range(ny):
                Phi0[i,jj]=Phi[i,jj]

        for i in range(1,nx-1):
            for j in range(1,ny-1):      
                    Phi[i,j]=(1.0-param)*Phi0[i,j]+param*(aW[i-1,j-1]*Phi[i-1,j]+aE[i-1,j-1]*Phi[i+1,j]+aS[i-1,j-1]*Phi[i,j-1]+aN[i-1,j-1]*Phi[i,j+1]+Sp[i-1,j-1])/aP[i-1,j-1]

        residual,m,Xm = Residual(Phi,nx-2,ny-2,aP,aE,aW,aN,aS,Sp,m,a,Xm)
        count_iter=count_iter+1

#        print(count_iter)
#        print(residual)        
#        end= time.process_time()
#        print(end-start)        
       
    return Phi,residual,m,Xm
#________________________
def Residual(Phi,ei,ej,aP,aE,aW,aN,aS,Sp,m,a,Xm):

    NINV = 1.0/(ei*ej)
    acum= np.zeros([ei,ej])

    for i in range(1,ei+1):  
        for j in range(1,ej+1):  
            acum[i-1,j-1]=aW[i-1,j-1]*Phi[i-1,j]+aE[i-1,j-1]*Phi[i+1,j]+aS[i-1,j-1]*Phi[i,j-1]+aN[i-1,j-1]*Phi[i,j+1]+Sp[i-1,j-1]-aP[i-1,j-1]*Phi[i,j]                
               
    residual= np.sqrt(NINV*np.sum(acum*acum))
    
    Res.append(residual)
    m=m+1
    nit.append(m)  
    t_par= time.process_time()
    end2=t_par-start1
    Tim.append(end2)

    if a==1:
        FRes.append(residual)
        Xm=Xm+1
        Fnit.append(Xm)  
        t_par= time.process_time()
        end2=t_par-start1
        FTim.append(end2)        

    else:
        TRes.append(residual)
        Xm=Xm+1
        Tnit.append(Xm)  
        t_par= time.process_time()
        end2=t_par-start1
        TTim.append(end2)        

    return residual,m,Xm
#________________________

#Grafica
#________________________

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

#Mallas
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

#________________________
def FT(xc,yc,u,v,Ae,Aw,An,As,dV,dt,Lambda,T,TT):

    aP=np.zeros([nx,ny]); aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
    aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny]);  
    Su=np.zeros([nx,ny]); Sp=np.zeros([nx,ny]);

    for i in range(1,nx+1):
        for jj in range(1,ny+1):
            dxe=xc[i+1]-xc[i]; dxw=xc[i]-xc[i-1]
            dyn=yc[jj+1]-yc[jj]; dys=yc[jj]-yc[jj-1]

            ue=u[i,jj]
            uw=u[i-1,jj]
            vn=v[i,jj]
            vs=v[i,jj-1]
           
            bE=Ae/dxe-ue*Ae/2.0
            bW=Aw/dxw+uw*Aw/2.0
            bN=An/dyn-vn*An/2.0
            bS=As/dys+vs*As/2.0
   
            #Neuman conditions
            if(i==1):
                bW=0 #West boundary, adiabatic
       
            if(i==nx):
                bE=0 #East boundary, adiabatic

            #Neuman conditions
#            if(j==1):
#                bS=0 #West boundary, adiabatic
       
#            if(j==ny):
#                bN=0 #East boundary, adiabatic
   
            bP=bE+bW+bN+bS+dV/dt
            Gamma=Lambda*(bP*TT[i,jj]-bE*TT[i+1,jj]-bW*TT[i-1,jj]-bN*TT[i,jj+1]-bS*TT[i,jj-1]-(dV/dt)*T[i,jj])
           
            aE[i-1,jj-1]=Ae/dxe
            aW[i-1,jj-1]=Aw/dxw
            aN[i-1,jj-1]=An/dyn
            aS[i-1,jj-1]=As/dys

    #Neuman conditions
            if(i==1):
                aW[i-1,jj-1]=0  #West boundary, adiabatic
           
            if(i==nx):
                aE[i-1,jj-1]=0  #East boundary, adiabatic

    #Neuman conditions
#            if(j==1):
#                aS[i-1,j-1]=0  #West boundary, adiabatic
           
#            if(j==ny):
#                aN[i-1,j-1]=0  #East boundary, adiabatic

            aP[i-1,jj-1]=aE[i-1,jj-1]+aW[i-1,jj-1]+aN[i-1,jj-1]+aS[i-1,jj-1]+dV/dt
            Sp[i-1,jj-1]=aP[i-1,jj-1]*TT[i,jj]-aE[i-1,jj-1]*TT[i+1,jj]-aW[i-1,jj-1]*TT[i-1,jj]-aN[i-1,jj-1]*TT[i,jj+1]-aS[i-1,jj-1]*TT[i,jj-1]-Gamma

    #BOUNDARY CONDITIONS

    #Top boundary: dirichlet
    Sp[:,ny-1]=Sp[:,ny-1]+aN[:,ny-1]*T[1:nx+1,ny+1]
    aN[:,ny-1]=0

    #Bottom boundary: dirichlet
    Sp[:,0]=Sp[:,0]+aS[:,0]*T[1:nx+1,0]
    aS[:,0]=0
   
    return aP,aE,aW,aN,aS,Sp
#________________________

def FF(x,y,u,Ae,Aw,An,As,dV,dt,Lambda,F,TT,Ra,alpha):

    aP=np.zeros([nx,ny]); aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
    aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny]);  
    Su=np.zeros([nx,ny]); Sp=np.zeros([nx,ny]);    
   
    for i in range(1,nx):
        for j in range(1,ny):
            dxe=x[i+1]-x[i]
            dxw=x[i]-x[i-1]
            dyn=y[j+1]-y[j]
            dys=y[j]-y[j-1]

            aE[i-1,j-1]=Ae/dxe
            aW[i-1,j-1]=Aw/dxw
            aN[i-1,j-1]=An/dyn
            aS[i-1,j-1]=As/dys
            aP[i-1,j-1]=aE[i-1,j-1]+aW[i-1,j-1]+aN[i-1,j-1]+aS[i-1,j-1]

#            Sp[i-1,j-1]=Ra*(math.cos(alpha)*(0.5*(TT[i+1,j]+TT[i+1,j+1])-0.5*(TT[i,j]+TT[i,j+1]))*Ae-math.sin(alpha)*(0.5*(TT[i,j+1]+TT[i+1,j+1])-0.5*(T[i,j]+TT[i+1,j]))*An)

            Sp[i-1,j-1]=Ra*((0.5*(TT[i+1,j]+TT[i+1,j+1])-0.5*(TT[i,j]+TT[i,j+1]))*Ae) #Tn*An-Ts*As

#Organizar los índices de los aPs          
    #East
    Sp[nx-2,0:ny-1]=Sp[nx-2,0:ny-1]+aE[nx-2,0:ny-1]*F[nx,1:ny]
    aE[nx-2,:]=0

    #West
    Sp[0,0:ny-1]=Sp[0,0:ny-1]+aW[0,0:ny-1]*F[0,1:ny]
    aW[0,:]=0

    #North
    Sp[0:nx-1,ny-2]=Sp[0:nx-1,ny-2]+aN[0:nx-1,ny-2]*F[1:nx,ny]
    aN[:,ny-2]=0

    #South
    Sp[0:nx-1,0]=Sp[0:nx-1,0]+aS[0:nx-1,0]*F[1:nx,0]
    aS[:,0]=0

    return aP,aE,aW,aN,aS,Sp

#________________________

def V_Field(x,y,u,v,Ae,Aw,An,As,dV,dt,Lambda,F,TT,Ra,nx,ny):

    for i in range(1,nx):
        for j in range(1,ny+1):
            u[i,j]=(F[i,j]-F[i,j-1])/dy

#    print(u)        

    for i in range(1,nx+1):
        for j in range(1,ny):
            v[i,j]=-(F[i,j]-F[i-1,j])/dx

#    print(v)        

    return u,v

def InterpolateToNodesUs(uc,us,ei,ej):

    for i in range(1,ei):
        for j in range(0,ej+1):
            uc[i,j]=(us[i-1,j]+us[i,j])*0.5

    uc[0,:]=us[0,:]
    uc[ei+1,:]=us[ei,:]


#================================================
def InterpolateToNodesVs(vc,vs,ei,ej):

    for i in range(0,ei+1):
        for j in range(1,ej):
            vc[i,j]=(vs[i,j]+vs[i,j-1])*0.5
           
    vc[:,0]=vs[:,0]
    vc[:,ej+1]=vs[:,ej]
#================================================

def Plot_UV(xc,yc,uc,vc):

    uc=np.transpose(uc)
    vc=np.transpose(vc)
    plt.quiver(xc, yc, uc, vc, color='g')
    plt.title('Vector Field')
 
#    plt.xlim(-7, 7)
#    plt.ylim(-7, 7)
 
    plt.grid()
    plt.show()

#________________________

#________________________
#________________________
#Programa principal

x,xc=Mallado(xc, x, x0, xl, nx)
y,yc=Mallado(yc, y, y0, yl, ny)

#ciclo temporal

for it in range(1,itmax+1):
    c=1
        #FIXED POINT: Maxval of the difference between the last and the present iterations is stored in deltaTemp
    deltaTemp=1.0
   
    while ((c < max_iter2) and (deltaTemp > 5.0E-6)):
       
#        print(it,c)        

        #Se resuelve la ecuación de Temperatura
        aP,aE,aW,aN,aS,Sp = FT(xc,yc,u,v,Ae,Aw,An,As,dV,dt,Lambda,T,TT)
        TT,res,m,Tm =Sor_2D(1.3,TT,nx+2,ny+2,aP,aE,aW,aN,aS,Sp,max_iter1,tolerance,residual,m,0,Tm)

        #Se resuelve la ecuación de Flujo
        aP,aE,aW,aN,aS,Sp = FF(x,y,u,Ae,Aw,An,As,dV,dt,Lambda,F,TT,Ra,alpha)
        F,r,m,Fm = Sor_2D(1.95,F,nx+1,ny+1,aP,aE,aW,aN,aS,Sp,max_iter1,tolerance,residual,m,1,Fm)

        deltaTemp=np.max(np.abs(convt[1:nx+1,1:ny+1]-TT[1:nx+1,1:ny+1]))        
        dTemp.append(deltaTemp)
        cTim.append(c)
        convt[1:nx+1,1:ny+1]=TT[1:nx+1,1:ny+1]

        u,v=V_Field(x,y,u,v,Ae,Aw,An,As,dV,dt,Lambda,F,TT,Ra,nx,ny)

        c=c+1

#________________________
    #Update T

    for i in range(nx+1):
            for jj in range(ny+1):
                T[i,jj]=TT[i,jj]

    Nu,Ng = Nusselt(nx,dx,dy,T)   
    Nu_g[it-1]=Ng
    tc[it-1]=it-1

InterpolateToNodesUs(uc,u,nx,ny)
InterpolateToNodesVs(vc,v,nx,ny)
   
T[0,:]=T[1,:]; T[nx+1,:]=T[nx,:]


plt.figure(1)
Plot_T(T,xc,yc,dx,dy,xl,yl,nx+2,ny+2)                 #Se grafican los resultados
plt.figure(2)
Plot_T(F,x,y,dx,dy,xl,yl,nx+2,ny+2)                 #Se grafican los resultados
plt.figure(3)
Plot_UV(xc,yc,uc,vc)
plt.figure(4)
plt.plot(tc,Nu_g)

plt.figure(5)
plt.plot(TTim,TRes)
plt.figure(6)
plt.plot(Tnit,TTim)
plt.figure(7)
plt.plot(FTim,FRes)
plt.figure(8)
plt.plot(Fnit,FTim)

end1= time.process_time()
print(end1-start1)
