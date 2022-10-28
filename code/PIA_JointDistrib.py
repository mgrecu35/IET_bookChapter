from netCDF4 import Dataset
import glob
import numpy as np
fs=glob.glob("data/colloc/*nc")
nt=0
piaKuL=[]
tbL=[]
kern=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

for f in fs:
    print(f)
    fh=Dataset(f)
    reliab=fh["reliabF"][:]
    reliabKu=fh["reliabKuF"][:]
    bzd=fh["bzd"][:]
    bcf=fh["bcf"][:]
    a1=np.nonzero((bzd-150)*(bzd-100)<0)
    dpia=fh["pia"][:]
    piaKu=fh["piaKu"][:]
    for i in a1[0]:
        if bcf[i]<bzd[i]+15:
            continue
        if (reliab[i]==1 or reliab[i]==2) and (reliabKu[i]==1 or reliabKu[i]==2):
            piaKuL.append([piaKu[i],dpia[i,0]])

import matplotlib.pyplot as plt
import matplotlib
piaKuL=np.array(piaKuL)
jointPIAS=plt.hist2d(piaKuL[:,0],piaKuL[:,1],bins=np.arange(40)*0.25,norm=matplotlib.colors.LogNorm(),cmap='jet')
import pickle
pickle.dump({"jointPIA_dPIA":jointPIAS},open("jointPIA.pklz","wb"))
