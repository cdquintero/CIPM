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

x0=0.0; lx=3.0
y0=0.0; ly=1.0                                       #Longitud

nxlh=[16]; nylh=[16]

dxlh=[(lx-x0)/(nxlh[0])]; dylh=[(ly-y0)/(nylh[0])]
dV=[dxlh[0]*dylh[0]];   dt=2.0E-4

x=[np.zeros(nxlh[0]+1)]; y=[np.zeros(nylh[0]+1)]
xc=[np.zeros(nxlh[0]+2)]; yc=[np.zeros(nylh[0]+2)]
u= [np.zeros([nxlh[0]+1,nylh[0]+2])]; v= [np.zeros([nxlh[0]+2,nylh[0]+1])]


Ae=[dylh[0]];  Aw=[dylh[0]]
An=[dxlh[0]];  As=[dxlh[0]]

Lambda=0.99

itmax=8000
max_iter1=2000 #solver
max_iter2=500 #fixed
tolerance1=5.0E-6
tolerance2=5.0E-6
Ra=100
alpha=0*(np.pi/180.0)

Tm=0
Fm=0
m=0
n=0 #counter to define steady state
n_mean=22000 #number of norms infinite of T to be averaged

residual=1
norm_inf=1.0 #norm infinite to define steady state

aWlh=[np.zeros([nxlh[0],nylh[0]])]; aElh=[np.zeros([nxlh[0],nylh[0]])]
aSlh=[np.zeros([nxlh[0],nylh[0]])]; aNlh=[np.zeros([nxlh[0],nylh[0]])]
aPlh=[np.zeros([nxlh[0],nylh[0]])]; Splh=[np.zeros([nxlh[0],nylh[0]])]

rTlh=[np.zeros([nxlh[0]+2,nylh[0]+2])]
rFlh=[np.zeros([nxlh[0]+1,nylh[0]+1])]
elh=[np.zeros([nxlh[0]+2,nylh[0]+2])]
eFlh=[np.zeros([nxlh[0]+1,nylh[0]+1])]

uc=np.zeros([nxlh[0]+2,nylh[0]+2])
vc=np.zeros([nxlh[0]+2,nylh[0]+2])

Flux=np.zeros([nxlh[0]+1,nylh[0]+1])
Flh=[np.zeros([nxlh[0]+1,nylh[0]+1])]
TTlh=[np.zeros([nxlh[0]+2,nylh[0]+2])]
Tlh=[np.zeros([nxlh[0]+2,nylh[0]+2])]

Tlh[0][:,0]=1.0
Tlh[0][:,nylh[0]+1]=0.0

TTlh[0][:,0]=1.0
TTlh[0][:,nylh[0]+1]=0.0

Nu=np.zeros(nxlh[0]+2)
Ng=0
tc=np.zeros(itmax)
Nu_g=np.zeros(itmax)


Res=[]
nit=[] 
Tim=[]

TRes=[]
FRes=[]
Tnit=[]
Fnit=[]
TTim=[]
FTim=[]

TResMM=[]
FResMM=[]
TnitMM=[]
FnitMM=[]
TTimMM=[]
FTimMM=[]

dTemp=[]
cTim=[]

#Ecuaciones discretizadas

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
def FT(xc,yc,u,v,Ae,Aw,An,As,dV,dt,Lambda,T,TT,nx,ny):

    aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
    aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny])  
    Sp=np.zeros([nx,ny]); aP=np.zeros([nx,ny])

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
def FF(x,y,u,v,Ae,Aw,An,As,dV,dt,Lambda,F,TT,Ra,alpha,nx,ny):

    aP=np.zeros([nx,ny]); aW=np.zeros([nx,ny]); aE=np.zeros([nx,ny])
    aS=np.zeros([nx,ny]); aN=np.zeros([nx,ny]);  
    Sp=np.zeros([nx,ny]);    
   
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

            Sp[i-1,j-1]=Ra*((0.5*(TT[i+1,j]+TT[i+1,j+1])-0.5*(TT[i,j]+TT[i,j+1]))*Ae) #Tn*An-Ts*As

    #Organizar los índices de los aPs          
    #East
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

def V_Field(x,y,dx,dy,u,v,Ae,Aw,An,As,dV,dt,Lambda,F,TT,Ra,nx,ny):

    for i in range(1,nx):
        for j in range(1,ny+1):
            u[i,j]=(F[i,j]-F[i,j-1])/dy

#    print(u)        

    for i in range(1,nx+1):
        for j in range(1,ny):
            v[i,j]=-(F[i,j]-F[i-1,j])/dx

#    print(v)        

    return u,v

#________________________

def SOR(param,aP,aS,aN,aW,aE,Su,nx,ny,f,max_iter):
 
    count_iter=0;        #Criterios de Convergencia
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

def SOR_2D(param,aP,aS,aN,aW,aE,Sp,nx,ny,f,max_iter):            
   
    tolerance=5.0E-6; count_iter=0; residual=1.0        #Criterios de Convergencia
    f0=np.zeros([nx+2,ny+2])
    rh=np.zeros([nx+2,ny+2])

    while count_iter < max_iter and residual > tolerance:  
       
        for i in range(nx+2):
            for jj in range(ny+2):
                f0[i,jj]=f[i,jj]

        for i in range(1,nx+1):
            for j in range(1,ny+1):        
                f[i,j]=(1.0-param)*f0[i,j]+param*(Sp[i-1,j-1]+aW[i-1,j-1]*f[i-1,j]+aE[i-1,j-1]*f[i+1,j]+aS[i-1,j-1]*f[i,j-1]+aN[i-1,j-1]*f[i,j+1])/aP[i-1,j-1]
 
        rh,residual = Residual(aP,aS,aN,aW,aE,Sp,nx,ny,f)

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

def Parametros(nl):

    for i in range(1,nl):

        nxlh.append(int(nxlh[i-1]/2))
        nylh.append(int(nylh[i-1]/2))
        dxlh.append((lx-x0)/(nxlh[i]))
        dylh.append((ly-y0)/(nylh[i]))

        x.append(np.zeros(nxlh[i]+1))
        y.append(np.zeros(nylh[i]+1))
        xc.append(np.zeros(nxlh[i]+2))
        yc.append(np.zeros(nylh[i]+2))
       
        x[i],xc[i]=Mallado(xc[i], x[i], x0, lx, nxlh[i])
        y[i],yc[i]=Mallado(yc[i], y[i], y0, ly, nylh[i])
   
        Ae.append(dylh[i])
        Aw.append(dylh[i])
        An.append(dxlh[i])
        As.append(dxlh[i])
        dV.append(dxlh[i]*dylh[i])

        u.append(np.zeros([nxlh[i]+1,nylh[i]+2]))
        v.append(np.zeros([nxlh[i]+2,nylh[i]+1]))
   
        aPlh.append(np.zeros([nxlh[i],nylh[i]]))
        aSlh.append(np.zeros([nxlh[i],nylh[i]]))
        aNlh.append(np.zeros([nxlh[i],nylh[i]]))
        aWlh.append(np.zeros([nxlh[i],nylh[i]]))
        aElh.append(np.zeros([nxlh[i],nylh[i]]))  
        Splh.append(np.zeros([nxlh[i],nylh[i]]))

        Flh.append(np.zeros([nxlh[i]+1,nylh[i]+1]))
   
        TTlh.append(np.zeros([nxlh[i]+2,nylh[i]+2]))
        TTlh[i][:,0]=1.0
        TTlh[i][:,nylh[i]+1]=0.0

        Tlh.append(np.zeros([nxlh[i]+2,nylh[i]+2]))
        Tlh[i][:,0]=1.0
        Tlh[i][:,nylh[i]+1]=0.0
           
        rTlh.append(np.zeros([nxlh[i]+2,nylh[i]+2]))
        rFlh.append(np.zeros([nxlh[i]+1,nylh[i]+1]))
        elh.append(np.zeros([nxlh[i]+2,nylh[i]+2]))                
        eFlh.append(np.zeros([nxlh[i]+1,nylh[i]+1]))                

#_______________________

def Restriction(n_r):
    
    for i in range (1,n_r):    

        r_aux = np.zeros([nxlh[i],nylh[i]])
               
     #Restricción              
        rTlh[i]=Prom(rTlh[i-1],nxlh[i],nylh[i])      

        aPlh[i],aElh[i],aWlh[i],aNlh[i],aSlh[i],Splh[i] = FT(xc[i],yc[i],u[i],v[i],Ae[i],Aw[i],An[i],As[i],dV[i],dt,Lambda,Tlh[i],TTlh[i],nxlh[i],nylh[i])  #Se calculan los coeficientes aW,aE,ap,Su,Sp

        for j in range(1,nxlh[i]+1):
            for k in range(1,nylh[i]+1):
                r_aux[j-1,k-1]=rTlh[i][j,k]

        elh[i]=SOR(1.3,aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],r_aux,nxlh[i],nylh[i],elh[i],10)      
    
        if i!=nl-1:
            for j in range(1,nylh[i]+1):
                for k in range(1,nxlh[i]+1):
                    rTlh[i][j][k] = rTlh[i][j][k]-(aPlh[i][j-1][k-1]*elh[i][j][k]-aSlh[i][j-1][k-1]*elh[i][j][k-1]-aNlh[i][j-1][k-1]*elh[i][j][k+1]-aWlh[i][j-1][k-1]*elh[i][j-1][k]-aElh[i][j-1][k-1]*elh[i][j+1][k])

#________________________

def Restriction_F(n_r):
   
    for i in range (1,n_r):    

        r_aux = np.zeros([nxlh[i]-1,nylh[i]-1])
               
     #Restricción              
        rFlh[i]=Prom(rFlh[i-1],nxlh[i]-1,nylh[i]-1)      

        aPlh[i],aElh[i],aWlh[i],aNlh[i],aSlh[i],Splh[i] = FF(x[i],y[i],u[i],v[i],Ae[i],Aw[i],An[i],As[i],dV[i],dt,Lambda,Flh[i],TTlh[i],Ra,alpha,nxlh[i],nylh[i])  #Se calculan los coeficientes aW,aE,ap,Su,Sp

        for j in range(1,nxlh[i]):
            for k in range(1,nylh[i]):
                r_aux[j-1,k-1]=rFlh[i][j,k]

        eFlh[i]=SOR(1.3,aPlh[i],aSlh[i],aNlh[i],aWlh[i],aElh[i],r_aux,nxlh[i]-1,nylh[i]-1,eFlh[i],10)      
    
        if i!=nl-1:
            for j in range(1,nylh[i]):
                for k in range(1,nxlh[i]):
                    rFlh[i][j][k] = rFlh[i][j][k]-(aPlh[i][j-1][k-1]*eFlh[i][j][k]-aSlh[i][j-1][k-1]*eFlh[i][j][k-1]-aNlh[i][j-1][k-1]*eFlh[i][j][k+1]-aWlh[i][j-1][k-1]*eFlh[i][j-1][k]-aElh[i][j-1][k-1]*eFlh[i][j+1][k])

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

def Prolongation(n_p):

    for i in range(1,n_p):

        elhp= np.zeros([nxlh[n_p-i-1]+2,nylh[n_p-i-1]+2])
        r_aux = np.zeros([nxlh[n_p-i-1],nylh[n_p-i-1]])

        elhp=Interpolation(elh[n_p-i],nxlh[n_p-i],nxlh[n_p-i-1],nylh[n_p-i],nylh[n_p-i-1])  
        elh[n_p-i-1]=elh[n_p-i-1]+elhp
    
        for j in range(1,nxlh[n_p-i-1]+1):
            for k in range(1,nylh[n_p-i-1]+1):
                r_aux[j-1,k-1]=rTlh[n_p-i-1][j,k]

        elh[n_p-i-1]=SOR(1.3,aPlh[n_p-i-1],aSlh[n_p-i-1],aNlh[n_p-i-1],aWlh[n_p-i-1],aElh[n_p-i-1],r_aux,nxlh[n_p-i-1],nylh[n_p-i-1],elh[n_p-i-1],2)      
#____________________________________________________________________

def Prolongation_F(n_p):

    for i in range(1,n_p):

        elhp= np.zeros([nxlh[n_p-i-1]+1,nylh[n_p-i-1]+1])
        r_aux = np.zeros([nxlh[n_p-i-1]-1,nylh[n_p-i-1]-1])

        elhp=Interpolation_F(eFlh[n_p-i],nxlh[n_p-i]-1,nxlh[n_p-i-1]-1,nylh[n_p-i]-1,nylh[n_p-i-1]-1)  
        eFlh[n_p-i-1]=eFlh[n_p-i-1]+elhp
    
        for j in range(1,nxlh[n_p-i-1]):
            for k in range(1,nylh[n_p-i-1]):
                r_aux[j-1,k-1]=rFlh[n_p-i-1][j,k]

        eFlh[n_p-i-1]=SOR(1.3,aPlh[n_p-i-1],aSlh[n_p-i-1],aNlh[n_p-i-1],aWlh[n_p-i-1],aElh[n_p-i-1],r_aux,nxlh[n_p-i-1]-1,nylh[n_p-i-1]-1,eFlh[n_p-i-1],2)      
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

#================================================
def Interpolation_F(e2lh,ei2h,eih,ej2h,ejh):    
 
    ehp=np.zeros([eih+2,ejh+2])  
   
    for i in range(1,ei2h+1):      
        for j in range(1,ej2h+1):
            ehp[2*i,2*j-1]=0.75*e2lh[i,j]+0.25*e2lh[i,j-1]
            ehp[2*i,2*j]=0.75*e2lh[i,j]+0.25*e2lh[i,j+1]

    for i in range(1,ei2h+1):      
        ehp[2*i,ejh]=0.75*e2lh[i,ej2h]


    for j in range(1,eih+1):      
        for i in range(1,ej2h+1):
            ehp[2*i-1,j]=0.75*ehp[2*i,j]+0.25*ehp[2*i-2,j]
            ehp[2*i,j]=0.75*ehp[2*i+2,j]+0.25*ehp[2*i,j]

    for j in range(1,eih+1):      
        ehp[eih,j]=0.75*ehp[eih-1,j]


    return ehp

#================================================

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
   
    plt.grid()
    plt.show()

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

#________________________
def R_norm(m,Xm,residual,a):
    
    Res.append(residual)
    m=m+1
    nit.append(m)  
    t_par= time.process_time()
    end2=t_par-start
    Tim.append(end2)

    if a==0:
        TResMM.append(residual)
        Xm=Xm+1
        TnitMM.append(Xm)
        t_par= time.process_time()
        end2=t_par-start
        TTimMM.append(end2)

    if a==1:    
        FResMM.append(residual)
        Xm=Xm+1
        FnitMM.append(Xm)
        t_par= time.process_time()
        end2=t_par-start
        FTimMM.append(end2)

    return (m,Xm)
#________________________

#Programa principal

x[0],xc[0]=Mallado(xc[0], x[0], x0, lx, nxlh[0])
y[0],yc[0]=Mallado(yc[0], y[0], y0, ly, nylh[0])
Parametros(nl)
convt= np.ones([nxlh[0]+2,nylh[0]+2])
residual=1
iter_MM=0

for it in range(1,itmax+1):
    c=1
   
    #FIXED POINT: Maxval of the difference between the last and the present iterations is stored in deltaTemp
    deltaTemp=1.0

    while ((c < max_iter2) and (deltaTemp > 5.0E-6)):

        aPlh[0],aElh[0],aWlh[0],aNlh[0],aSlh[0],Splh[0] = FT(xc[0],yc[0],u[0],v[0],Ae[0],Aw[0],An[0],As[0],dV[0],dt,Lambda,Tlh[0],TTlh[0],nxlh[0],nylh[0])  #Se calculan los coeficientes aW,aE,ap,Su,Sp
        TTlh[0] = SOR(1.3,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0],nylh[0],TTlh[0],3)
        rTlh[0],residual = Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0],nylh[0],TTlh[0])
        
        m,Tm=R_norm(m,Tm,residual,0)

        while iter_MM < 10000 and residual >5.0E-6:  
            TTlh[0] = SOR(1.3,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0],nylh[0],TTlh[0],5)
            rTlh[0],residual = Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0],nylh[0],TTlh[0])

            Restriction(nl)
            Prolongation(nl)            
            TTlh[0]=TTlh[0]+elh[0]        

            TTlh[0] = SOR(1.3,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0],nylh[0],TTlh[0],2)

            m,Tm=R_norm(m,Tm,residual,0)

            for i in range (nl):
                elh[i]=np.zeros([int(nxlh[i]+2),int(nylh[i]+2)])

            iter_MM=iter_MM+1  

        iter_MM=0

        aPlh[0],aElh[0],aWlh[0],aNlh[0],aSlh[0],Splh[0] = FF(x[0],y[0],u[0],v[0],Ae[0],Aw[0],An[0],As[0],dV[0],dt,Lambda,Flh[0],TTlh[0],Ra,alpha,nxlh[0],nylh[0])  #Se calculan los coeficientes aW,aE,ap,Su,Sp
        Flh[0] = SOR(1.95,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0],3)
        rFlh[0],residual = Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0])

        m,Fm=R_norm(m,Fm,residual,1)


#        while iter_MM < 10000 and residual >5.0E-6:  
#
 #           Flh[0] = SOR(1.95,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0],5)
  #          rFlh[0],residual = Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0])

   #         Restriction_F(nl)
    #        Prolongation_F(nl)            
                
     #       Flh[0]=Flh[0]+eFlh[0]        
                
      #      Flh[0] = SOR(1.95,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0],2)
       #     rFlh[0],residual = Residual(aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0])

        #    m,Fm=R_norm(m,Fm,residual,1)

         #   for i in range (nl):
          #      eFlh[i]=np.zeros([int(nxlh[i]+1),int(nylh[i]+1)])

           # iter_MM=iter_MM+1  

       # iter_MM=0
        
        Flh[0] = SOR_2D(1.95,aPlh[0],aSlh[0],aNlh[0],aWlh[0],aElh[0],Splh[0],nxlh[0]-1,nylh[0]-1,Flh[0],max_iter1)

        u[0],v[0] = V_Field(x[0],y[0],dxlh[0],dylh[0],u[0],v[0],Ae[0],Aw[0],An[0],As[0],dV[0],dt,Lambda,Flh[0],TTlh[0],Ra,nxlh[0],nylh[0])

        deltaTemp=np.max(np.abs(convt[1:nxlh[0]+1,1:nylh[0]+1]-TTlh[0][1:nxlh[0]+1,1:nylh[0]+1]))  
        dTemp.append(deltaTemp)
        cTim.append(c)

        convt[1:nxlh[0]+1,1:nylh[0]+1]=TTlh[0][1:nxlh[0]+1,1:nylh[0]+1]
   
    c=c+1
#________________________
    #Update T
   
    for i in range(nxlh[0]+1):
            for jj in range(nylh[0]+1):
                Tlh[0][i,jj]=TTlh[0][i,jj]

    Nu,Ng = Nusselt(nxlh[0],dxlh[0],dylh[0],Tlh[0])   
    Nu_g[it-1]=Ng
    tc[it-1]=it-1
    
#________________________

Tlh[0][0,:]=Tlh[0][1,:]; Tlh[0][nxlh[0]+1,:]=Tlh[0][nxlh[0],:]  

InterpolateToNodesUs(uc,u[0],nxlh[0],nylh[0])
InterpolateToNodesVs(vc,v[0],nxlh[0],nylh[0])

for i in range(nxlh[0]+1):
    for jj in range(nylh[0]+1):
        Flux[i,jj]=Flh[0][i,jj]



plt.figure(1)
Plot_T(Tlh[0],xc[0],yc[0],dxlh[0],dylh[0],x[0],y[0],nxlh[0]+2,nylh[0]+2)                 #Se grafican los resultados
plt.figure(2)
Plot_T(Flh[0],x[0],y[0],dxlh[0],dylh[0],x[0],y[0],nxlh[0]+2,nylh[0]+2)
plt.figure(3)
Plot_UV(xc[0],yc[0],uc,vc)
plt.figure(4)
plt.plot(tc,Nu_g)

plt.figure(5)
plt.plot(TTimMM,TResMM)
plt.figure(6)
plt.plot(TnitMM,TTimMM)
plt.figure(7)
plt.plot(FTimMM,FResMM)
plt.figure(8)
plt.plot(FnitMM,FTimMM)

end1= time.process_time()
print(end1-start)
