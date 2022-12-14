import matplotlib.pyplot as plt
from netCDF4 import Dataset
nmfreq=8
nmu=5
import numpy as np
import multiscatterLib as mscatterlib

#stop


fDPR='../case/2A-CS-KWAJ.GPM.DPR.V9-20211125.20150311-S162503-E162626.005866.V07A.HDF5'


fhD=Dataset(fDPR)
lond=fhD["FS/Longitude"][:]
latd=fhD["FS/Latitude"][:]
zsfc=fhD["FS/SLV/zFactorFinalESurface"][:]
zm=fhD["FS/PRE/zFactorMeasured"][:]
bcf=fhD["FS/PRE/binClutterFreeBottom"][:]
bzd=fhD["FS/VER/binZeroDeg"][:]
piaF=fhD["FS/SLV/piaFinal"][:]
srt_pia=fhD["FS/SRT/pathAtten"][:]
bcf=fhD['FS/PRE/binClutterFreeBottom']

zm=np.ma.array(zm,mask=zm<0)
binZeroDeg=fhD["FS/VER/binZeroDeg"][:]

bzd=binZeroDeg[21,16]-1
n3=np.array([bzd-4,bzd,bzd+2])
#plt.scatter(zm[21,16,:,0][n3],n3)

from lkTables import *

lkT=scattTables()
betaR= 0.7116278
alphaR=10**(-3.19939512)
betaS=1.0346435
alphaS=10**(-5.29043193)

dnw=np.zeros((176),float)
def hb(zKum,dn,alpha,beta,dr):
    q=0.2*np.log(10)
    zeta=q*beta*dn**(1-beta)*alpha*10**(0.1*zKum*beta)*dr
    zetaSum=zeta.cumsum()[-1]
    if zetaSum>0.995:
        f=0.995/zetaSum
    else:
        f=1
    zc=zKum-10/beta*np.log10(1-f*zeta.cumsum())
    pia=-10/beta*np.log10(1-f*zeta.cumsum())
    return zc,pia

dr=0.125
bcf1=bcf[21,16]

#plt.plot(zc1[:bcf1],range(0,bcf1))


def getZKa_snow(dnw,zc,lkT):
    n=zc.shape[0]
    zka=np.zeros((n),float)-99
    for k in range(n):
        if zc[k]>12:
            ibin=int((zc[k]-10*dnw[k]+12)/0.25)
            if ibin<=0:
                ibin=0
                dnw[k]=(zc[k]+12)/10.
            if ibin>=252:
                ibin=0
                dnw[k]=(zc[k]-lkT.zKaS[252])/10.
            zka[k]=lkT.zKaS[ibin]+10*dnw[k]
    return zka



def getZKa_graup(dnw,zc,lkT,dr):
    n=zc.shape[0]
    zka=np.zeros((n),float)-99
    pRate=np.zeros((n),float)-99
    piaKa=0
    for k in range(n):
        if zc[k]>12:
            ibin=int((zc[k]-10*dnw[k]+12)/0.25)
            if ibin<=0:
                ibin=0
                dnw[k]=(zc[k]+12)/10.
            if ibin>=271:
                ibin=0
                dnw[k]=(zc[k]-lkT.zKaG[271])/10.
            zka[k]=lkT.zKaG[ibin]+10*dnw[k]
            piaKa+=lkT.attKaG[ibin]*dr*10**(dnw[k])
            zka[k]-=piaKa
            piaKa+=lkT.attKaG[ibin]*dr*10**(dnw[k])
            pRate[k]=lkT.graupRate[ibin]*dr*10**(dnw[k])
    return zka,piaKa,pRate

def getZKa_snow(dnw,zc,lkT,dr):
    n=zc.shape[0]
    zka=np.zeros((n),float)-99
    zka_true=np.zeros((n),float)-99
    pRate=np.zeros((n),float)-99
    piaKa=0
    kextKa=np.zeros((n),float)+1e-4
    salbKa=np.zeros((n),float)
    asymKa=np.zeros((n),float)
    for k in range(n):
        if zc[k]>12:
            ibin=int((zc[k]-10*dnw[k]+12)/0.25)
            if ibin<=0:
                ibin=0
                dnw[k]=(zc[k]+12)/10.
            if ibin>=252:
                ibin=0
                dnw[k]=(zc[k]-lkT.zKaS[252])/10.
            zka[k]=lkT.zKaS[ibin]+10*dnw[k]
            zka_true[k]=lkT.zKaS[ibin]+10*dnw[k]
            piaKa+=lkT.attKaS[ibin]*dr*10**(dnw[k])
            zka[k]-=piaKa
            piaKa+=lkT.attKaS[ibin]*dr*10**(dnw[k])
            pRate[k]=lkT.snowRate[ibin]*dr*10**(dnw[k])
            kextKa[k]=lkT.kextS[ibin,3]*10**dnw[k]
            salbKa[k]=lkT.salbS[ibin,3]
            asymKa[k]=lkT.asymS[ibin,3]
    return zka,zka_true,piaKa,pRate,kextKa,salbKa,asymKa



def getZKa_rain(dnw,zc,lkT,dr,piaKa):
    n=zc.shape[0]
    zka=np.zeros((n),float)-99
    zka_true=np.zeros((n),float)-99
    pRate=np.zeros((n),float)-99
    kextKa=np.zeros((n),float)+1e-4
    salbKa=np.zeros((n),float)
    asymKa=np.zeros((n),float)
    for k in range(n):
        if zc[k]>12:
            zkaR,attKaR,pRateR,kextKaR,salbKaR,asymKaR=getRainProp(zc[k],dnw[k],lkT)
            if k<2:
                zkaS,attKaS,pRateS,kextKaS,salbKaS,asymKaS=getSnowProp(zc[k],dnw[k],lkT)
                fract=(k+1)/3.
                zkam=10.0*np.log10(fract*10**(0.1*zkaR)+(1-fract)*10**(0.1*zkaS))
                attKa=fract*attKaR+(1-fract)*attKaS
                pRatem=fract*pRateR+(1-fract)*pRateS
                #print(zkaG,zkaR)
                #print(kextKaS,salbKaS,asymKaS)
                kextKa[k]=fract*kextKaR+(1-fract)*kextKaS
                salbKa[k]=fract*kextKaR*salbKaR+(1-fract)*kextKaS*salbKaS
                asymKa[k]=fract*kextKaR*salbKaR*asymKaR+(1-fract)*kextKaS*salbKaS*asymKaS
                asymKa[k]/=salbKa[k]
                salbKa[k]/=kextKa[k]
            else:
                zkam=zkaR
                attKa=attKaR
                pRatem=pRateR
                kextKa[k]=kextKaR
                salbKa[k]=salbKaR
                asymKa[k]=asymKaR
            ibin=int((zc[k]-10*dnw[k]+12)/0.25)
            zka[k]=zkam
            zka_true[k]=zkam
            piaKa+=attKa*dr
            zka[k]-=piaKa
            piaKa+=attKa*dr
            pRate[k]=pRatem
            #if k<2:
            #    print(zka[k]+piaKa,piaKa)
    return zka,zka_true,piaKa,pRate,kextKa,asymKa,salbKa


dr=0.125

zm1=zm[21,16,:,0]
dnw[0:bzd]=(bzd-1-np.arange(bzd))*0.02+0.5
dnw+=0.5
zc1=zm1.copy()

def ret1D(zm1,bzd,alphaS,betaS,alphaR,betaR,dr,lkT,dnw):
    zc1=zm1.copy()
    zcs,pias=hb(zm1[0:bzd],10**dnw[0:bzd],alphaS,betaS,dr)
    zc1[0:bzd]=zcs
    zcr,pia=hb(zm1[bzd+1:bcf1]+pias[-1],10**dnw[bzd+1:bcf1],alphaR,betaR,dr)
    zc1[bzd+1:bcf1]=zcr
    zkaG,zkaG_true,piaG,pRateG,kextKaG,salbKaG,asymKaG=getZKa_snow(dnw,zcs,lkT,dr)
    zkaR,zkaR_true,piaKaR,pRateR,kextKaR,asymKaR,salbKaR=getZKa_rain(dnw[bzd:],zcr,lkT,dr,piaG)
    zkaG=np.ma.array(zkaG,mask=zkaG<0)
    zkaR=np.ma.array(zkaR,mask=zkaR<0)
    zka_sim=np.concatenate((zkaG,zkaR))
    zka_sim=np.ma.array(zka_sim,mask=zka_sim<0)
    return zka_sim,piaKaR,kextKaR,asymKaR,salbKaR,kextKaG,salbKaG,asymKaG,zkaG_true,zkaR_true


zka_sim,pia_sim,kextKaR,asymKaR,salbKaR,\
    kextKaG,salbKaG,asymKaG,zkaG_true,zkaR_true=ret1D(zm1,bzd,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
kext=np.concatenate((kextKaG,kextKaR))
salb=np.concatenate((salbKaG,salbKaR))
asym=np.concatenate((asymKaG,asymKaR))

dr=0.125
alt=400
freq=35.5
nonorm=0
ztrue=np.concatenate((zkaG_true,zkaR_true))
ist=80
noms=0
theta=0.5
zms = mscatterlib.multiscatterf(kext[ist:],salb[ist:],asym[ist:],ztrue[ist:],dr,noms,alt,theta,freq,nonorm)

plt.figure(figsize=(6,6))
plt.plot(zm[21,16,:,0],range(176))
plt.plot(zm[21,16,:,1],range(176))
plt.ylim(175,80)
plt.plot(zka_sim,range(zka_sim.shape[0]))
plt.xlabel('dBZ')
plt.ylabel('Range bin')
plt.legend(['Z$_{obs}$(Ku)','Z$_{obs}$(Ka)','Z$_{sim}$(Ka)'])
plt.savefig('retrievedZKa.png')
np.random.seed(10)
x1=np.random.randn(20)
x1=np.exp(0.125*x1)
x1/=x1.mean()
dz=np.log10(x1)*10.0
zka_av=np.zeros((zka_sim.shape[0]),float)
nc=0
zkaL=[]
piaKaL=[]
for dz1 in dz:
    zka_sim,pia_sim,kextKaR,asymKaR,salbKaR,\
        kextKaG,salbKaG,asymKaG,zkaG_true,zkaR_true=ret1D(zm1+dz1,bzd,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
    print(pia_sim)
    zka_av+=10**(0.1*zka_sim.data)
    #if pia_sim>100:
    #    stop
    zkaL.append(zka_sim.data)
    piaKaL.append(pia_sim)
    nc+=1

#
zms=np.ma.array(zms,mask=zms<0)                
zka_av=np.log10(zka_av/nc)*10
zka_av=np.ma.array(zka_av,mask=zka_av<0)
plt.plot(zka_av,range(zka_sim.shape[0]))
plt.plot(zms,range(ist,kext.shape[0]))
plt.legend(['Z$_{obs}$(Ku)','Z$_{obs}$(Ka)','Z$_{sim}$(Ka)','Z$_{NUBFsim}$(Ka)','Z$_{MS}$(Ka)'])
plt.savefig('retrievedZKa.png')

piaKa_av=np.log10((10**(-0.1*np.array(piaKaL))).mean())*10
