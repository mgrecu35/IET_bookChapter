from netCDF4 import Dataset
import glob
import numpy as np
fs=glob.glob("../data/colloc/*nc")
nt=0
piaKuL=[]
tbL=[]
kern=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
nt2=0

for f in fs:
    print(f)
    fh=Dataset(f)
    reliab=fh["reliabF"][:]
    bzd=fh["bzd"][:]
    bcf=fh["bcf"][:]
    a1=np.nonzero((bzd-150)*(bzd-100)<0)
    a2=np.nonzero(reliab[a1]==1)
    piaKu=fh["pia"][:]
    zKu=fh["zKu"][:]
    zKuL=[]
    zKu[zKu<0]=0
    tb=fh["Tb"][:,1::2,1::2,:]
    tb2=fh["Tb"][:,2:5,2:5,:]
    X=[]
    y=[]
    for i in a1[0]:
        if bcf[i]<bzd[i]+15:
            continue
        if not(reliab[i]==1 or reliab[i]==2):
            continue
        x1=[]
        x1=[(kern@tb[i,:,:,k]).sum() for k in range(9)]
        x1.extend(tb2[i,1,1,:])
        x1.extend(zKu[i,bzd[i]-30:bzd[i]+15:2])
        if piaKu[i,1]<20:
            X.append(x1)
            y.append(piaKu[i])
        else:
            if np.random.random()<0.2:
                X.append(x1)
                y.append(piaKu[i])
    tbL.extend(X)
    piaKuL.extend(y)
    nt+=len(a2[0])
    reliab=fh["reliabKuF"][:]
    dreliab=fh["reliabF"][:]
    dpiaKu=fh["pia"][:]
    piaKu=fh["piaKu"][:]
    a2=np.nonzero(abs(reliab[a1]-1.5)*0.6)
    X=[]
    y=[]
    for i in a1[0][a2]:
        if bcf[i]<bzd[i]+15:
            continue
        if (dreliab[i]==1 or dreliab[i]==2) and dpiaKu[i,1]<8:
            continue
        x1=[]
        if dreliab[i]==1 or dreliab[i]==2:
            x1=[(kern@tb[i,:,:,k]).sum() for k in range(9)]
            x1.extend(tb2[i,1,1,:])
            x1.extend(zKu[i,bzd[i]-30:bzd[i]+15:2])
            X.append(x1)
            y.append([piaKu[i],piaKu[i]*6])
    nt2+=len(X)
    tbL.extend(X)
    piaKuL.extend(y)

#stop
from sklearn import neighbors
k=50
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(\
    np.array(tbL)[:,:], np.array(piaKuL), test_size=0.33, random_state=42)
n_neighbors = k



from tensorflow import keras
from keras.layers import Dense,BatchNormalization,Dropout
nin=X_train.shape[1]
model = keras.models.Sequential()
model.add(Dense(8,input_shape=(nin,),activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8,activation="relu"))
#model.add(Dropout(0.25))
model.add(Dense(8,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(2,activation="linear"))
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalery = StandardScaler()

X_trainNN=scalerX.fit_transform(X_train[:,:])
y_trainNN=scalery.fit_transform(y_train[:,:])
X_testNN=scalerX.transform(X_test[:,:])
y_testNN=scalery.transform(y_test[:,:])

print(scalerX.mean_)
print(scalerX.scale_)
print(scalery.mean_)
print(scalery.scale_)

import pickle
pickle.dump({"scalerX":scalerX,"scalery":scalery},open("scalers.pklz","wb"))
model.fit(X_trainNN, y_trainNN, epochs=40, batch_size=64)
model.save('piaTbConvZ_with_SF_SRT.h5')

import pickle

d=pickle.load(open("jointPIA.pklz","rb"))
jointPIA=d["jointPIA_dPIA"]

y_=model.predict(X_testNN)
pia_=scalery.inverse_transform(y_)

import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
ax=plt.subplot(122)
piaHist2d=plt.hist2d(y_test[:,0],pia_[:,0],bins=np.arange(40)*0.25,norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_aspect('equal')
plt.xlabel('Reference PIA(Ku)[dB]')
plt.ylabel('ML PIA(Ku)[dB]')
plt.xlim(0,7)
plt.ylim(0,7)
cbar=plt.colorbar(piaHist2d[-1],orientation='horizontal')
cbar.ax.set_title('Counts')

ax=plt.subplot(121)
c=plt.pcolormesh(np.arange(40)*0.25,np.arange(40)*0.25,jointPIA[0],norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_aspect('equal')
plt.xlabel('dSRT PIA(Ku)[dB]')
plt.ylabel('SRT PIA(Ku)[dB]')
plt.xlim(0,7)
plt.ylim(0,7)
cbar=plt.colorbar(c,orientation='horizontal')
cbar.ax.set_title('Counts')
plt.tight_layout()
plt.savefig('ML_PIA.png')
