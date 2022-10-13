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


fh_gmi=Dataset("case/1C-R-CS-KWAJ.GPM.GMI.XCAL2016-C.20150311-S162548-E162730.005866.V07A.HDF5")
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
#plt.figure()
#ax = plt.axes(projection=ccrs.PlateCarree())
#plt.pcolormesh(lond,latd,zsfc[:,:,0],cmap='jet',vmin=10,vmax=50)
#plt.xlim(lond.min(),lond.max())
#plt.ylim(latd.min(),latd.max())
#plt.colorbar()
import numpy as np
zm=np.ma.array(zm,mask=zm<0)
fig=plt.figure(figsize=(8,6))
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
plt.plot(range(zm.shape[0]),tc_regrid[:,15,1])
#plt.plot(range(zm.shape[0]),tc_regrid[:,15,5])
plt.xlim(0,zm.shape[0])
plt.legend(["10.65-GHz V"])
plt.ylabel("Brightness temperature [K]")
plt.xlabel('Relative scan number')


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
