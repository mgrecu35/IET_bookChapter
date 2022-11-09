import matplotlib.pyplot as plt
from netCDF4 import Dataset
nmfreq=8
nmu=5

import pickle

d=pickle.load(open("scalers_32.pklz","rb"))
from tensorflow import keras
model = keras.models.load_model('piaTbZ_with_SF_SRT.h5')

fDPR='../case/2A-CS-KWAJ.GPM.DPR.V9-20211125.20150311-S162503-E162626.005866.V07A.HDF5'
f2AKu='../case/2A-CS-KWAJ.GPM.Ku.V9-20211125.20150311-S162503-E162626.005866.V07A.HDF5'

import pyresample

fhD=Dataset(fDPR)
fhKu=Dataset(f2AKu)
lond=fhD["FS/Longitude"][:]
latd=fhD["FS/Latitude"][:]
zsfc=fhD["FS/SLV/zFactorFinalESurface"][:]
zm=fhD["FS/PRE/zFactorMeasured"][:]
bcf=fhD["FS/PRE/binClutterFreeBottom"][:]
bzd=fhD["FS/VER/binZeroDeg"][:]
pRateDPR=fhD["FS/SLV/precipRate"][:]
bbPeak=fhD["FS/CSF/binBBPeak"][:,:]
zm0=zm.copy()
zm0[zm0<0]=0
piaF=fhD["FS/SLV/piaFinal"][:]
srt_pia=fhD["FS/SRT/pathAtten"][:]
srt_piaKu=fhKu["FS/SRT/pathAtten"][:]

fh_gmi=Dataset("../case/1C-R-CS-KWAJ.GPM.GMI.XCAL2016-C.20150311-S162548-E162730.005866.V07A.HDF5")
gmi_lon=fh_gmi["S1/Longitude"][:]
gmi_lat=fh_gmi["S1/Latitude"][:]
tc=fh_gmi["S1/Tc"][:]

from pyresample import kd_tree, geometry
swath_def = geometry.SwathDefinition(lons=gmi_lon, lats=gmi_lat)
target_def = geometry.SwathDefinition(lons=lond[:,:], lats=latd[:,:])
tc_regrid = kd_tree.resample_gauss(swath_def, tc[:,:,:],
                                   target_def, radius_of_influence=25000, \
                                   sigmas=[12500 for k in range(9)])
    
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

X=[]
for i in range(zm.shape[0]):
    x1=[]
    x1.extend(tc_regrid[i,15,:])
    x1.extend(zm0[i,15,bzd[i,15]-30:bzd[i,15]+15:2,0])
    X.append(x1)

X=np.array(X)
scalerX=d["scalerX"]
scalery=d["scalery"]
X_NN=scalerX.transform(X)
y_=model.predict(X_NN)
pia_nn=scalery.inverse_transform(y_)
iplot=1
if iplot==1:
    ax4=plt.subplot(211)
    ax4.plot(range(zm.shape[0]),pia_nn[:,1]-pia_nn[:,0])
    ax4.plot(range(zm.shape[0]),piaF[:,15,1]-piaF[:,15,0])
    ax4.plot(range(zm.shape[0]),srt_pia[:,15,1]-srt_pia[:,15,0])
    ax4.plot(range(zm.shape[0]),srt_piaKu[:,15]*5)
    ax4.set_ylabel("dPIA [dB]")
    plt.xlim(0,90)


dr=0.125
from radarRetrSubs import *

piaHB1=[]
piaHB2=[]
pRateL=[]
nsfcRate1=[]
nsfcRate=[]
nsfcRate2=[]
for i in range(0,100):
    dnw=np.zeros((176),float)
    zm1=zm[i,15,:,0]
    bzd1=bzd[i,15]
    dnw[0:bzd1]=(bzd1-1-np.arange(bzd1))*0.02+0.05
    dnw+=0.3
    bcf1=bcf[i,16]
    zc1=zm1.copy()
    if bbPeak[i,15]<=0:
        zka_sim,pia_sim2,kextKaR,asymKaR,salbKaR,\
            kextKaG,salbKaG,asymKaG,\
            zkaG_true,zkaR_true,pRate=ret1D(zm1,bzd1,bcf1,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
    else:
        zka_sim,pia_sim2,kextKaR,asymKaR,salbKaR,\
            kextKaG,salbKaG,asymKaG,\
            zkaG_true,zkaR_true,pRate=ret1Dst(zm1,bzd1,bbPeak[i,15],bcf1,alphaS,betaS,alphaR,\
                                              betaR,dr,lkT,dnw)
        #print(bzd1,bbPeak[i,15])
    pRate[pRate<0]=0
   
    piaHB2.append(pia_sim2)
    nsfcRate2.append(pRate[-1])

    dnw=np.zeros((176),float)
    zm1=zm[i,15,:,0]
    bzd1=bzd[i,15]
    dnw[0:bzd1]=(bzd1-1-np.arange(bzd1))*0.02+0.05
    dnw+=-0.3
    bcf1=bcf[i,16]
    zc1=zm1.copy()
    if bbPeak[i,15]<=0:
        zka_sim,pia_sim1,kextKaR,asymKaR,salbKaR,\
            kextKaG,salbKaG,asymKaG,\
            zkaG_true,zkaR_true,pRate=ret1D(zm1,bzd1,bcf1,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
    else:
        zka_sim,pia_sim1,kextKaR,asymKaR,salbKaR,\
            kextKaG,salbKaG,asymKaG,\
            zkaG_true,zkaR_true,pRate=ret1Dst(zm1,bzd1,bbPeak[i,15],bcf1,alphaS,betaS,alphaR,\
                                              betaR,dr,lkT,dnw)
        #print(bzd1,bbPeak[i,15])
    pRate[pRate<0]=0
    piaHB1.append(pia_sim1)
    dndpia=(pia_sim2-pia_sim1)/0.6
    dns=2
    if bbPeak[i,15]==0:
        pia_nn[i,1]=0.25*pia_nn[i,1]+0.75*5*srt_piaKu[i,15]
    dn=(pia_nn[i,1]-0.5*(pia_sim1+pia_sim2))*dns*dndpia/(dns*dndpia**2+9)
    if pia_nn[i,1]>50:
        print(dn)
        if dn>0.75:
            dn=0.75
    #print(dn)
    nsfcRate1.append(pRate[-1])
    dnw=np.zeros((176),float)
    zm1=zm[i,15,:,0]
    bzd1=bzd[i,15]
    dnw[0:bzd1]=(bzd1-1-np.arange(bzd1))*0.02+0.05
    #if bbPeak[i,15]==0:
    #    print(dn)
        #and dn<0.2:
        #dn=0.5
    dnw+=dn
    bcf1=bcf[i,16]
    zc1=zm1.copy()
    if bbPeak[i,15]<=0:
        zka_sim,pia_sim1,kextKaR,asymKaR,salbKaR,\
            kextKaG,salbKaG,asymKaG,\
            zkaG_true,zkaR_true,pRate=ret1D(zm1,bzd1,bcf1,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
    else:
        zka_sim,pia_sim1,kextKaR,asymKaR,salbKaR,\
            kextKaG,salbKaG,asymKaG,\
            zkaG_true,zkaR_true,pRate=ret1Dst(zm1,bzd1,bbPeak[i,15],bcf1,alphaS,betaS,alphaR,\
                                              betaR,dr,lkT,dnw)
        #print(bzd1,bbPeak[i,15])
    pRate[pRate<0]=0
    pRateL.append(pRate[:166])
    nsfcRate.append(pRate[-1])

plt.fill_between(range(100), 5/6*np.array(piaHB1),5/6*np.array(piaHB2),alpha=0.2)
ax4.legend(["ML","DPR","dSRT","SRT","HB"],loc="upper right")
#plt.yscale('log')
plt.ylim(1,100)
plt.subplot(212)
plt.plot(nsfcRate)
plt.fill_between(range(100),nsfcRate1,nsfcRate2,alpha=0.2)
#plt.plot(pRateDPR[0:100,15,-4])
plt.xlim(0,90)
plt.ylabel("Precipitation Rate (mm/h)")
plt.xlabel("Relative scan")
plt.savefig("retPIA.png")
plt.figure()

import matplotlib
plt.pcolormesh(np.array(pRateL[0:90]).T[::-1,:],\
               norm=matplotlib.colors.LogNorm(vmin=0.1,vmax=50),cmap='jet')
plt.colorbar()

plt.figure()
import matplotlib
plt.pcolormesh(np.array(pRateDPR[1:90,15,:]).T[::-1,:],\
               norm=matplotlib.colors.LogNorm(vmin=0.1,vmax=50),cmap='jet')
plt.colorbar()
