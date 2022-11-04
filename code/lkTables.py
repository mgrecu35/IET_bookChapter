from netCDF4 import Dataset
class scattTables:
    fh=Dataset("../lookupTables/scatteringTablesGPM.nc")
    zKuR=fh["zKuR"][:289]
    zKaR=fh["zKaR"][:289]
    dmr=fh["dmr"][:289]
    rainRate=fh["rainRate"][:289]
    attKuR=fh["attKuR"][:289]
    attKaR=fh["attKaR"][:289]
    dmr=fh["dmr"][:289]
    #--------------------#
    zKuS=fh["zKuS"][:253]
    zKaS=fh["zKaS"][:253]
    dms=fh["dms"][:253]
    snowRate=fh["snowRate"][:253]
    attKuS=fh["attKuS"][:253]
    attKaS=fh["attKaS"][:253]
    #------------------#
    dmg=fh["dmg"][:272]
    zKuG=fh["zKuG"][:272]
    zKaG=fh["zKaG"][:272]
    dmg=fh["dmg"][:272]
    graupRate=fh["graupRate"][:272]
    attKuG=fh["attKuG"][:272]
    attKaG=fh["attKaG"][:272]
    dmg=fh["dmg"][:272]

    
def getGraup(zc,dnw,lkT):
    if zc>12:
        ibin=int((zc-10*dnw+12)/0.25)
        if ibin<=0:
            ibin=0
            dnw=(zc+12)/10.
        if ibin>=271:
            ibin=0
            dnw=(zc-lkT.zKaG[271])/10.
        zka=lkT.zKaG[ibin]+10*dnw
        attKa=lkT.attKaG[ibin]*10**dnw
        pRate=lkT.graupRate[ibin]*10**dnw
    else:
        zka=-99
        attKa=0
        pRate=0
    return zka,attKa,pRate

def getRain(zc,dnw,lkT):
    if zc>12:
        ibin=int((zc-10*dnw+12)/0.25)
        if ibin<=0:
            ibin=0
            dnw=(zc+12)/10.
        if ibin>=288:
            ibin=0
            dnw=(zc-lkT.zKaR[271])/10.
        zka=lkT.zKaR[ibin]+10*dnw
        attKa=lkT.attKaR[ibin]*10**dnw
        pRate=lkT.rainRate[ibin]*10**dnw
    else:
        zka=-99
        attKa=0
        pRate=0
    return zka,attKa,pRate
        
