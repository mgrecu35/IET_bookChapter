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
    bzd=fh["bzd"][:]
    bcf=fh["bcf"][:]
    a1=np.nonzero((bzd-150)*(bzd-100)<0)
    a2=np.nonzero(reliab[a1]==1)
    piaKu=fh["pia"][:][a1][a2]
    zKu=fh["zKu"][:]
    zKuL=[]
    zKu[zKu<0]=0
    tb=fh["Tb"][:,1::2,1::2,:]
    tb2=fh["Tb"][:,2:5,2:5,:]
    X=[]
    for i in a1[0][a2]:
        if bcf[i]<bzd[i]+15:
            continue
        #x1=[(kern@tb[i,:,:,k]).sum() for k in range(9)]
        #x1.extend([(kern@tb2[i,:,:,k]).sum() for k in range(9)])
        x1=[]
        x1.extend(tb2[i,1,1,:])
        x1.extend(zKu[i,bzd[i]-30:bzd[i]+15:2])
        X.append(x1)
    #stop
    
    #zKuL=np.array(zKuL)[:,:]
    #tb=(fh["Tb"][:,2:5,2:5,:][a1][a2]).reshape(len(a2[0]),3*3*9)
    #X=np.append(tb,zKuL,axis=1)
    tbL.extend(X)
    piaKuL.extend(piaKu)
    nt+=len(a2[0])
    reliab=fh["reliabKuF"][:]
    dreliab=fh["reliabF"][:]
    dpiaKu=fh["pia"][:]
    piaKu=fh["piaKu"][:]
    a2=np.nonzero(reliab[a1]==1)
    X=[]
    y=[]
    for i in a1[0][a2]:
        if bcf[i]<bzd[i]+15:
            continue
        if (dreliab[i]==1 or dreliab[i]==2) and dpiaKu[i,1]<15:
            continue
        #x1=[(kern@tb[i,:,:,k]).sum() for k in range(9)]
        #x1.extend([(kern@tb2[i,:,:,k]).sum() for k in range(9)])
        x1=[]
        x1.extend(tb2[i,1,1,:])
        x1.extend(zKu[i,bzd[i]-30:bzd[i]+15:2])
        X.append(x1)
        y.append([piaKu[i],piaKu[i]*6])
    #stop
    
    #zKuL=np.array(zKuL)[:,:]
    #tb=(fh["Tb"][:,2:5,2:5,:][a1][a2]).reshape(len(a2[0]),3*3*9)
    #X=np.append(tb,zKuL,axis=1)
    #tbL.extend(X)
    #piaKuL.extend(y)
    #stop

from sklearn import neighbors
k=50
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(\
    np.array(tbL)[:,:], np.array(piaKuL), test_size=0.33, random_state=42)
n_neighbors = k
#knn_sk = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
#knn_sk.fit(X_train, y_train)
#y_=knn_sk.predict(X_test)


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

model.fit(X_trainNN, y_trainNN, epochs=40, batch_size=64)
model.save('piaTbZ_noSF_SRT.h5')
