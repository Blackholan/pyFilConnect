import numpy as np
from astropy.io import fits
from sklearn.neighbors import KDTree
import os


print('version of feb, 3, 2023')

class ReadDTFE:
    #--------------------------------------------------------#
    # DOCUMENT:                                              #
    # Reads the ascii file produced with netconv from a      #
    # NDnet file.                                            #
    # The structure contains the following field:            #
    # ndim: number of dimensions                             #
    # x0: origin of the bounding box                         #
    # delta: extent of the bounding box                      #
    # nvert: number of vertices                              #
    # posvert: position of the vertices (2D or 3D)           #
    # nseg: number of segments                               #
    # nfaces: number of faces                                #
    # nvolumes: number of volumes                            #
    # idseg: id of vertices corresponding to segments        #
    # idfaces: id of vertices corresponding to faces         #
    # idvolumes: id of vertices corresponding to volumes     #
    # fieldvalue: density at the location                    #
    # ++ if opt='all': the entire file is read and stored    #
    # ++ if opt='wght': the entire file is read but only the #
    # info for weighting are stored;                         # 
    # ++ if opt='visu': the entire file is read but only the #
    # info for visualisation are stored;                     #
    # ++ if opt='short':only header info are read            #
    #--------------------------------------------------------#
    # EXAMPLE:                                               #
    # delaunay_3D cluster_0160.asc                           #
    # netconv  cluster_0160.asc.NDnet -to NDnet_ascii        #
    # NDNet=ReadDTFE('cluster_0160.asc.NDnet.a.NDNet',opt=1) #
    #--------------------------------------------------------#
    #
    def __init__(self,ndnet,opt='all'):
        try: 
            with open(ndnet, "r")  as f:    
                # read general info -------------------------#
                tmp=f.readline()                             #
                ndim=int((f.readline()).split("\n")[0])      # ndims
                tmp=f.readline()                             #
                tmp=f.readline()                             #
                self.ndim=ndim                               #
                x0l=np.array(((tmp.split("[")[1]).split("]")[0].split(","))).astype(np.float32)
                self.x0=x0l                                  #
                deltal=np.array(tmp.split("[")[2].split("]")[0].split(",")).astype(np.float32)
                self.delta=deltal                            #
                nvert=int((f.readline()).split("\n")[0])     # nvertices
                self.nvert=nvert  
                if ((opt=='all')+(opt=='wght')+(opt=='visu')):
                    posvert=np.zeros((ndim,nvert))               # vertex positions
                    posvert=posvert.astype(np.float32)           #
                    for i in range(nvert):                       #
                        tmp=f.readline()                         #
                        posvert[:,i]=np.array((tmp.split("\n")[0]).split(" ")).astype(np.float32)
                    self.posvert=posvert                         #
                    #--------------------------------------------#
                    #
                    # read segment ------------------------------#
                    tmp=f.readline()       
                    dimseg=int(((tmp).split("\n")[0]).split(" ")[0])+1 # elt type (segment)
                    nseg=int(((tmp).split("\n")[0]).split(" ")[1])     # number segments
                    self.nseg=nseg
                    if ((opt=='all')):
                        idseg=np.zeros((dimseg,nseg))            # vertex id for seg
                        idseg=idseg.astype(int)                  #
                    for i in range(nseg):                        #
                        tmp=f.readline()                         #
                        if ((opt=='all')):
                            idseg[:,i]=np.array((tmp.split("\n")[0]).split(" ")).astype(np.int32)
                    if ((opt=='all')):    
                        self.idseg=idseg
                    #--------------------------------------------#
                    #
                    # read tetrahedron faces --------------------#
                    tmp=f.readline()                             #
                    dimseg=int(((tmp).split("\n")[0]).split(" ")[0])+1 # elt type (face)
                    nseg=int(((tmp).split("\n")[0]).split(" ")[1])     # number faces
                    self.nfaces=nseg
                    if ((opt=='all')+(ndim==2)*(opt=='visu')):
                        idfac=np.zeros((dimseg,nseg))            # vertex id for faces
                        idfac=idfac.astype(int)                  #
                    for i in range(nseg):                        #
                        tmp=f.readline()                         #
                        if ((opt=='all')+(ndim==2)*(opt=='visu')):         #
                            idfac[:,i]=np.array((tmp.split("\n")[0]).split(" ")).astype(np.int32)
                    if ((opt=='all')+(ndim==2)*(opt=='visu')):             #
                        self.idfaces=idfac
                    #--------------------------------------------#
                    #
                    # read tetrahedron volumes in 3d ------------#
                    if (ndim==3):                                #
                        tmp=f.readline()                         #
                        dimseg=int(((tmp).split("\n")[0]).split(" ")[0])+1# elt type (tetrahedron)
                        nseg=int(((tmp).split("\n")[0]).split(" ")[1])  # number tet
                        if ((opt=='all')+(opt=='visu')): 
                            idvol=np.zeros((dimseg,nseg))        # vertex id for tet
                            idvol=idvol.astype(int)              #
                        for i in range(nseg):                    #
                            tmp=f.readline()                     #
                            if ((opt=='all')+(opt=='visu')): 
                                idvol[:,i]=np.array((tmp.split("\n")[0]).split(" ")).astype(np.int32)
                        self.nvolumes=nseg
                        if ((opt=='all')+(opt=='visu')): 
                            self.idvolumes=idvol
                    #
                           
                    # read field values -------------------------#
                    if ((opt=='all')+(opt=='wght')+(opt=='visu')):  
                        tmp=f.readline()                         #
                        tmp=f.readline() ##read field values     #
                        tmp=f.readline()                         #
                        fieldvalue=np.zeros(nvert)               #
                        fieldvalue=fieldvalue.astype(np.float32) #
                        for i in range(nvert):                   #
                            tmp=f.readline()                     #
                            fieldvalue[i]=tmp                    #
                        self.fieldvalue=fieldvalue               #
                    #--------------------------------------------#
        except IOError as err:
                  print ("Cannot read ",ndnet)
    
class ReadSKL:
    #---------------------------------------------------#
    # DOCUMENT:                                         #
    # Reads the ascii file produced with skelconv from a#
    # NDskl file.                                       #
    # The structure contains the following field:       #
    #                                                   #
    # ndim: number of dimensions                        #
    # x0: origin of the bounding box                    #
    # delta: extent of the bounding box                 #
    #                                                   #
    #[CRITICAL POINTS]                                  #
    # ncrit: number of critical points                  #
    # Then, for each crit. point:                       #
    # type, pos, value, ID, boundary_flag               #
    # nfil: number of filaments connected to this CP    #
    # Then, for each filament connected to this CP      #
    # idcp,idfil: index of the CP at the end of this ...#
    # ...fil and index of the filament                  #
    #                                                   #
    #[FILAMENTS]                                        #
    # nfil: total number of filaments                   #
    # Then, for each filaments:                         #
    # CP1, CP2, nseg: index of CPs and nbr of segments  #
    # Positions of the  sampling points on the fil      #
    #                                                   #
    #[CRITICAL POINTS DATA]                             #
    # NF: nbr of fields associated to each CP           #
    # Then, for each field:                             #
    # Field name                                        #
    # Then, for each crit. point:                       #
    # v1,v2,v3, ..: value of each field                 #
    #                                                   #
    #[FILAMENT DATA]                                    #
    # NF: nbr of fields associated to each point of ... #
    # ...each filament                                  #
    # Then, for each filament:                          #
    # Then, for each point of the filament:             #
    # v1,v2,v3, ..: value of each field at the point    #
    #                                                   #
    #---------------------------------------------------#
    # EXAMPLE:                                          #
    #---------------------------------------------------#
    #
    def __init__(self,ndskl):
        try: 
            with open(ndskl, "r")  as f:
                
                # do a first passage to determin nsegmax and nfilmax--#
                g=open(ndskl, "r")
                tmp=g.readline() 
                ndim=int((g.readline()).split("\n")[0])
                tmp=g.readline()                        #
                tmp=g.readline()                        #
                tmp=g.readline()                        #
                ncrit=int((g.readline()).split("\n")[0])#
                nfilmax=0
                for i in range(ncrit):                  #
                    tmp=g.readline()                    #
                    nfil=int((g.readline()).split("\n")[0]) 
                    if (nfil>nfilmax):
                        nfilmax=nfil
                    for j in range(nfil):
                        tmp=g.readline()
                tmp=g.readline()                        #
                nfil=int((g.readline()).split("\n")[0]) # 
                nsegmax=0
                for i in range(nfil):                   #
                    tmp=g.readline()                    #
                    tmpp2=np.array((tmp.split("\n")[0]).split(" ")).astype(np.float)
                    nseg=np.int(tmpp2[2])
                    if (nseg>nsegmax):
                        nsegmax=nseg
                    for j in range(nseg):
                        tmp=g.readline()
                g.close()
                
                # read general info --------------------#
                tmp=f.readline()                 
                ndim=int((f.readline()).split("\n")[0]) # ndims
                self.ndim=ndim   ###                  
                tmp=f.readline()                        #
                tmp=f.readline()                        #
                x0l=np.array((tmp.split("[")[1]).split("]")[0].split(",")).astype(np.float)
                self.x0=x0l  ###
                deltal=np.array((tmp.split("[")[2].split("]")[0].split(","))).astype(np.float)
                self.delta=deltal  ###
                # read crit. points --------------------#
                tmp=f.readline()                        #
                ncrit=int((f.readline()).split("\n")[0])# number of critical points
                self.NCP=ncrit  ###
                poscrit=np.zeros([ndim,ncrit])          # critical points positions
                typecrit=np.zeros(ncrit)                # critical points type
                indxcrit=np.zeros(ncrit)                # critical points indices
                valucrit=np.zeros(ncrit)                # critical points values
                flagcrit=np.zeros(ncrit)                # critical points boundary_flag
                nfilcrit=np.zeros(ncrit)                # critical points nbr of filaments
                indfcrit=np.zeros([nfilmax,ncrit])      # indices of fil. connected to this CP
                for i in range(ncrit):                  #
                    tmp=f.readline()                    #
                    tmpp2=np.array((tmp.split("\n")[0]).split(" ")).astype(np.float)
                    typecrit[i]=np.int(tmpp2[0])
                    poscrit[:,i]=tmpp2[1:1+ndim]
                    valucrit[i]=tmpp2[1+ndim]
                    indxcrit[i]=np.int(tmpp2[2+ndim])
                    flagcrit[i]=np.int(tmpp2[3+ndim])              
                    nfil=int((f.readline()).split("\n")[0]) 
                    nfilcrit[i]=nfil
                    for j in range(nfil):
                        tmp=f.readline()
                        tmpp2=(np.array((tmp.split("\n")[0]).split(" "))[1:]).astype(np.float)
                        indfcrit[:nfil,i]=tmpp2[1]
                self.CPindf=indfcrit ###
                self.CPnfil=nfilcrit ###
                self.CPflag=flagcrit ###
                self.CPindx=indxcrit ###
                self.CPvalu=valucrit ###
                self.CPtype=typecrit ###
                self.CPposs=poscrit  ###
                # read filaments -----------------------#
                tmp=f.readline()                        #
                nfil=int((f.readline()).split("\n")[0]) # 
                self.NFIL=nfil  ###
                FILcp1=np.zeros(nfil)
                FILcp2=np.zeros(nfil)
                FILnsg=np.zeros(nfil)
                FILpos=np.zeros([nsegmax,ndim,nfil])     
               
                for i in range(nfil):                   #
                    tmp=f.readline()                    #
                    tmpp2=np.array((tmp.split("\n")[0]).split(" ")).astype(np.float)
                    FILcp1[i]=np.int(tmpp2[0])
                    FILcp2[i]=np.int(tmpp2[1])
                    FILnsg[i]=np.int(tmpp2[2])
                    nseg=FILnsg[i]
                    for j in range(np.int(nseg)):
                        tmp=f.readline()
                        FILpos[j,:,i]=(np.array((tmp.split("\n")[0]).split(" "))[1:]).astype(np.float)
                self.FILcp1=FILcp1
                self.FILcp2=FILcp2
                self.FILnsg=FILnsg
                self.FILpos=FILpos
                #---------------------------------------#
                tmp=f.readline()
                NFIELDS=int((f.readline()).split("\n")[0]) 
                self.CPnfld=NFIELDS
                CPname=np.chararray(NFIELDS,itemsize=20)
                for i in range(NFIELDS):
                    CPname[i]=f.readline().split("\n")[0]
                self.CPname=CPname
                CPvfld=np.zeros([NFIELDS,self.NCP])
                for i in range(self.NCP):    
                    tmp=f.readline()          
                    tmpp2=np.array((tmp.split("\n")[0]).split(" ")).astype(np.float)
                    CPvfld[:,i]=tmpp2
                self.CPvfld=CPvfld
                #---------------------------------------#
                tmp=f.readline()
                NFIELDS=int((f.readline()).split("\n")[0])
                self.FILnfd=NFIELDS
                FILnam=np.chararray(NFIELDS,itemsize=20)
                for i in range(NFIELDS):
                    FILnam[i]=f.readline().split("\n")[0]
                self.FILnam=FILnam
                FILvfd=np.zeros([NFIELDS,nsegmax,self.NFIL])
                for i in range(self.NFIL):
                    nseg=FILnsg[i]
                    for j in range(np.int(nseg)):
                        tmp=f.readline()
                        #print(tmp)
                        tmpp2=np.array((tmp.split("\n")[0]).split(" ")).astype(np.float)
                        #print(tmpp2)
                        FILvfd[:,j,i]=tmpp2
                self.FILvfd=FILvfd
                
        except IOError as err:
               print ("Cannot read ",ndskl)
    

def VOLUME(x,y,z):
    if (len(z)==0):  
        return abs(x[0,:]*y[1,:]-x[1,:]*y[0,:])*0.5
    else:
        return abs((x[1,:]*y[2,:]-x[2,:]*y[1,:])*z[0,:]-
          (x[0,:]*y[2,:]-x[2,:]*y[0,:])*z[1,:]+
          (x[0,:]*y[1,:]-x[1,:]*y[0,:])*z[2,:])/6.



def FILE4WEIGHTING(ndnet,outname,field4weight,posinit,logscale=0, opt_field=0):
    #---------------------------------------------------#
    # DOCUMENT:                                         #
    # Write the weighting file to feed to  netconv      #
    # in order to weight the tesselation                #
    # posinit is a ascii file containing particle positions
    # field4weight is the corresponding weights (e.g. mass) 
    # outname is the name of the output fits file       #
    #---------------------------------------------------#
    # EXAMPLE:                                          #
    # delaunay_3d cluster.asc -btype smooth             #
    # netconv  cluster.asc.NDnet -to NDnet_ascii        #
    # FILE4WEIGHTING('cluster.asc.NDnet.a.NDnet','weight.txt','mass.txt','pos.asc')
    # netconv cluster.asc.NDnet -addField weight.txt field_value
    #---------------------------------------------------#
    #
    NDnet=ReadDTFE(ndnet,opt='wght')
    print('tesselation read')
    if (opt_field ==0):
        mass=np.loadtxt(field4weight, skiprows=2)
        posinit=np.loadtxt(posinit, skiprows=2)
    else:
        mass=np.loadtxt(field4weight)
        posinit=np.loadtxt(posinit)
        
    nhal=np.shape(posinit)[0]
    
    ##check if the vertices are well ordered based on their positions

    newmass=np.zeros(NDnet.nvert)
    posvert=np.transpose(NDnet.posvert)
    kdt = KDTree(posinit, leaf_size=1000, metric='euclidean')
    dist, ind=kdt.query(posvert[:nhal,:], k=1, return_distance=True) 
    #if (logscale==1):
    #    mass=np.log10(mass)
    newmass[:nhal]=mass[ind[:,0]]
    
    if (NDnet.nvert>nhal):
        print ("there are more vertices than halos. It is ok")
        newmass[nhal:]=np.mean(mass[ind])
    if (logscale==1):
        newmass=np.log10(newmass*NDnet.fieldvalue)#/1e-33)
    else:
        newmass=newmass*NDnet.fieldvalue#/(1e-27)
        
    np.savetxt(outname,newmass,header='ANDFIELD\n['+str(NDnet.nvert)+']',comments='')
    #Now write the outfput file
    #f=open(outname,'w')
    #f.write("ANDFIELD")
    #f.write("\n")
    #f.write("["+str(NDnet.nvert)+"]")
    #f.write("\n")
    #for i in range(NDnet.nvert):
    #    f.write(str((newmass[i]/np.mean(mass))))
    #    f.write("\n")
    #f.close()
    #print (outname+" has been written")
    #return
    
def DTFE2GRID(ndnet,N,field4weight=[],posinit=[],outname='',z=3,opt_field=0):
    #---------------------------------------------------#
    # DOCUMENT:                                         #
    # Reads the ascii file produced with netconv from a #
    # NDnet file and interpolate the tesselation on a   #
    # cartesian grid.                                   #
    # ndnet is the name of the ascii file,              #
    # outname is the name of the output fits file       #
    # N is the required number of pixels along each     #
    # dimension.                                        #
    #---------------------------------------------------#
    # EXAMPLE:                                          #
    # delaunay_3d cluster.asc                           #
    # netconv  cluster.asc.NDnet -to NDnet_ascii        #
    # grid=DTFE2GRID('cluster.asc.NDnet.a.NDNet',"out.fits",[100,100,100])  
    #---------------------------------------------------#
    #
    msol=2*1e34
    mpc=(3.086*1e24/0.704*1./(1+z))
    NDnet=ReadDTFE(ndnet,opt='visu')
    #pprint('the tesselation has been read')
    pos=NDnet.posvert
    ndims=NDnet.ndim
    nvert=NDnet.nvert
    if (ndims==2):
        tet=NDnet.idfaces     #list of vertices in each tetrahedron
    else:
        tet=NDnet.idvolumes 
    x0=NDnet.x0
    delta=NDnet.delta
    xmax=x0+delta
    ntet=np.shape(tet)[1]
    tetPos=np.zeros((ndims,ndims+1,ntet))
    for i in range(ndims+1):
        for j in range(ndims):
            tetPos[j,i,:]=pos[j,tet[i]]
    a=tetPos[:,1,:]-tetPos[:,0,:];
    b=tetPos[:,2,:]-tetPos[:,0,:];
    if (ndims==3):
        c=tetPos[:,3,:]-tetPos[:,0,:]
    else:
        c=[]
    vol=abs(VOLUME(a,b,c)) # volume of each tetrahedron  
    w=np.where(vol==0)[0]
    w2=np.where(vol>0)[0]
    if (len(w)>0):
        vol[w]=np.min(vol[w2])
    vertexVol=np.zeros(nvert)
    vertexField=np.zeros(nvert)
    #####
    if (len(field4weight)>0):
        if (opt_field==0):
            mass=np.loadtxt(field4weight, skiprows=2)
            posinit=np.loadtxt(posinit, skiprows=2)
        else:
            mass=np.loadtxt(field4weight)
            posinit=np.loadtxt(posinit)
            
        nhal=np.shape(posinit)[0]
        ##check if the vertices are well ordered based on their positions
        newmass=np.zeros(NDnet.nvert)
        posvert=np.transpose(NDnet.posvert)
        kdt = KDTree(posinit, leaf_size=1000, metric='euclidean')
        dist, ind=kdt.query(posvert[:nhal,:], k=1, return_distance=True) 
        newmass[:nhal]=mass##[ind[:,0]]
        if (NDnet.nvert>nhal):
            newmass[nhal:]=np.min(mass[ind])
        tt=NDnet.fieldvalue
        field=(tt)*newmass
    else:
        field=NDnet.fieldvalue
    #print('ok weighting')
    #tetField=np.zeros(ntet)
  
    #for i in range(ntet):
    #    tetField[i]=np.mean(field[tet[:,i]]) #average field over each tetrahedron  
    tetField=np.mean(field[tet],axis=0)
    vertexField=field*0.
    
    #for i in range(ntet):
    #    idl=tet[:,i];v=vol[i];          
    #    vertexField[idl]=vertexField[idl]+v*tetField[i]
    #    vertexVol[idl]=vertexVol[idl]+v
        
    nt=np.tile(vol*tetField,(np.shape(tet)[0],1))
    nt2=np.tile(vol,(np.shape(tet)[0],1))
    
    vertexField[tet]+=nt
    vertexVol[tet]+=nt2
    vertexField/=vertexVol
    #####
    #print('okok1')
    x=np.arange(x0[0],xmax[0],delta[0]/(N[0]+1))
    y=np.arange(x0[1],xmax[1],delta[1]/(N[1]+1))
    if (ndims==3):
        z=np.arange(x0[2],xmax[2],delta[2]/(N[2]+1))
    else:
        z=[]
    x=(x[1:]+x[:-1])/2.
    y=(y[1:]+y[:-1])/2.

    #print('okok2')
    if (ndims==3):
      z=(z[1:]+z[:-1])/2.
      result=np.zeros((N[0],N[1],N[2]))
      xx=np.zeros((N[0],N[1],N[2]))
      yy=np.zeros((N[0],N[1],N[2]))
      zz=np.zeros((N[0],N[1],N[2]))
      
      for i in range(max(N)):
          if (i<N[0]):
              xx[:,:,i]=i
          if (i<N[1]):
              yy[:,i,:]=i
          if (i<N[2]):
              zz[i,:,:]=i
      xx=xx.flatten()
      yy=yy.flatten()
      zz=zz.flatten()
      gpos=np.zeros((ndims,(N[0])*(N[1])*(N[2])))
      gpos[0,:]=x[xx.astype(np.int32)]
      gpos[1,:]=y[yy.astype(np.int32)]
      gpos[2,:]=z[zz.astype(np.int32)]
    else:
      result=np.zeros((N[0],N[1]))
      xx=np.zeros((N[0],N[1]))
      yy=np.zeros((N[0],N[1]))
      for i in range(max(N)):
          if (i<N[0]):
              xx[i,:]=i
          if (i<N[1]):
              yy[:,i]=i
      result=result.flatten()
      xx=xx.flatten()
      yy=yy.flatten()
      gpos=np.zeros((ndims,(N[0])*(N[1])))
      gpos[0,:]=x[xx.astype(np.int32)]
      gpos[1,:]=y[yy.astype(np.int32)]
  
    pos=np.transpose(pos)
    gpos=np.transpose(gpos)
    #print ("Compute Tree")
    kdt = KDTree(pos, leaf_size=1000, metric='euclidean')
    dist, ind=kdt.query(gpos, k=1, return_distance=True)  
    result=vertexField[ind]#*msol/mpc/mpc**2
        
    if (ndims==3):
      result=result.reshape((N[0],N[1],N[2]))
    else:
      result=result.reshape((N[0],N[1]))
    if (len(outname)>0):
        hdu = fits.PrimaryHDU(result)
        hdul = fits.HDUList([hdu])
        os.system('rm -f '+outname)
        hdul.writeto(outname)
    return result
