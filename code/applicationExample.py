import matplotlib.pyplot as plt
from netCDF4 import Dataset
nmfreq=8
nmu=5

import pickle

d=pickle.load(open("scalers_32.pklz","rb"))
from tensorflow import keras
model = keras.models.load_model('piaTbZ_with_SF_SRT.h5')
#piaTbConvZ_with_SF_SRT
#print(model.summary())
#stop
#rteLib.readtablesliang2(nmu,nmfreq)
fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20220311-S235430-E235603.045647.V07A.HDF5'
fDPR='../case/2A-CS-KWAJ.GPM.DPR.V9-20211125.20150311-S162503-E162626.005866.V07A.HDF5'
#fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20150310-S040452-E040612.005842.V07A.HDF5'
#fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20150307-S050703-E050832.005796.V07A.HDF5'
#fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20150303-S183930-E184101.005743.V07A.HDF5'
import pyresample

fhD=Dataset(fDPR)
lond=fhD["FS/Longitude"][:]
latd=fhD["FS/Latitude"][:]
zsfc=fhD["FS/SLV/zFactorFinalESurface"][:]
zm=fhD["FS/PRE/zFactorMeasured"][:]
bcf=fhD["FS/PRE/binClutterFreeBottom"][:]
bzd=fhD["FS/VER/binZeroDeg"][:]
zm0=zm.copy()
zm0[zm0<0]=0
piaF=fhD["FS/SLV/piaFinal"][:]
srt_pia=fhD["FS/SRT/pathAtten"][:]

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
    zm=np.ma.array(zm,mask=zm<0)
    fig=plt.figure(figsize=(8,8))
    ax1=plt.subplot(311)
    ax11=plt.pcolormesh(zm[:,15,:,0].T,vmin=10,vmax=45,cmap='jet')
    plt.ylabel('Range bin')
    plt.title('Ku-band')
    ax1.xaxis.set_visible(False)
    plt.ylim(175,80)
    cb=plt.colorbar()
    cb.ax.set_title('dBZ')
    
    ax2=plt.subplot(312)
    plt.pcolormesh(zm[:,15,:,1].T,vmin=10,vmax=40,cmap='jet')
    ax2.xaxis.set_visible(False)
    plt.ylabel('Range bin')
    plt.ylim(175,80)
    #plt.xlabel('Relative scan number')
    plt.title('Ka-band')
    plt.colorbar()
    plt.tight_layout()
    x0=ax1.get_position().x0
    x1=ax1.get_position().x1
    ax3 = fig.add_axes([x0, 0.1, 0.720, 0.205])
    #plt.plot(range(zm.shape[0]),tc_regrid[:,15,0])
    ax3.plot(range(zm.shape[0]),tc_regrid[:,15,2])
    ax3.plot(range(zm.shape[0]),tc_regrid[:,15,3])
    #plt.plot(range(zm.shape[0]),tc_regrid[:,15,5])
    ax3.set_xlim(0,zm.shape[0])
    ax3.legend(["18.7-GHz V","18.7-GHz H"])
    ax3.set_ylabel("Brightness temperature [K]")
    ax3.set_xlabel('Relative scan number')
    if 1==1:
        ax4 = ax3.twinx()
        ax4.plot(range(zm.shape[0]),pia_nn[:,1]-pia_nn[:,0])
        ax4.plot(range(zm.shape[0]),piaF[:,15,1]-piaF[:,15,0])
        ax4.plot(range(zm.shape[0]),srt_pia[:,15,1]-srt_pia[:,15,0])
        ax4.legend(["CMB","DPR","dSRT"],loc="lower right")
        ax4.set_ylabel("dPIA [dB]")
    plt.savefig('kwajObs.20150311-S162503-E162626.005866.png')

    plt.figure(figsize=(8,6))
    
    plt.subplot(121)
    plt.plot(zm[20,19,:,0],range(176))
    plt.plot(zm[20,19,:,1],range(176))
    plt.ylim(175,75)
    plt.xlim(10,50)
    plt.xlabel('dBZ')
    plt.subplot(122)
    plt.plot(zm[35,19,:,0],range(176))
    plt.plot(zm[35,19,:,1],range(176))
    plt.ylim(175,75)
    plt.xlabel('dBZ')
    plt.xlim(10,50)
