import matplotlib.pyplot as plt
from netCDF4 import Dataset
nmfreq=8
nmu=5
#rteLib.readtablesliang2(nmu,nmfreq)
fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20220311-S235430-E235603.045647.V07A.HDF5'
fDPR='case/2A-CS-KWAJ.GPM.DPR.V9-20211125.20150311-S162503-E162626.005866.V07A.HDF5'
#fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20150310-S040452-E040612.005842.V07A.HDF5'
#fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20150307-S050703-E050832.005796.V07A.HDF5'
#fDPR='2A-CS-KWAJ.GPM.DPR.V9-20211125.20150303-S183930-E184101.005743.V07A.HDF5'
import pyresample

fhD=Dataset(fDPR)
lond=fhD["FS/Longitude"][:]
latd=fhD["FS/Latitude"][:]
zsfc=fhD["FS/SLV/zFactorFinalESurface"][:]
zm=fhD["FS/PRE/zFactorMeasured"][:]

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(lond,latd,zsfc[:,:,0],cmap='jet',vmin=10,vmax=50)
plt.xlim(lond.min(),lond.max())
plt.ylim(latd.min(),latd.max())
plt.colorbar()
import numpy as np
zm=np.ma.array(zm,mask=zm<0)
plt.figure()
ax1=plt.subplot(211)
plt.pcolormesh(zm[:,15,:,0].T,vmin=10,vmax=45,cmap='jet')
plt.ylabel('Range bin')
plt.title('Ku-band')
ax1.xaxis.set_visible(False)
plt.ylim(175,80)
cb=plt.colorbar()
cb.set_title('dBZ')

plt.subplot(212)
plt.pcolormesh(zm[:,15,:,1].T,vmin=10,vmax=40,cmap='jet')
plt.ylabel('Range bin')
plt.ylim(175,80)
plt.xlabel('Relative scan number')
plt.title('Ka-band')
plt.colorbar()
