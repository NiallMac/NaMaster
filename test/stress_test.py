import numpy as np
import healpy as hp
#import matplotlib.pyplot as plt
import sys
import pymaster as nmt
import os

np.random.seed(1234)

if len(sys.argv)!=8 :
    print "python stress_test.py nside nsims fsky aposize nholes rholes contaminants"
    exit(1)

nside=int(sys.argv[1])
nsims=int(sys.argv[2])
fsky=float(sys.argv[3])
aposize=float(sys.argv[4])
nholes=int(sys.argv[5])
rholes=float(sys.argv[6])
contaminants=int(sys.argv[5])

alpha_cont=0.1
use_cont=False
if contaminants>0 :
    use_cont=True

predir="tests_ns%d"%(nside)
os.system("mkdir -p "+predir)
prefix=predir+"/run_ns%d_fs%.2lf_apo%.2lf_nh%d_rh%.2lf"%(nside,fsky,aposize,nholes,rholes)
if use_cont :
    prefix+="_wcont"

#C_ell
r_theta=1/(np.sqrt(3.)*nside)
def window_pixel(l) :
    x=l*r_theta
    f=0.532+0.006*(x-0.5)**2
    y=f*x*1.0
    return np.exp(-y**2/2)
lmax=3*nside-1
tilt=-1.0
l0=100.
larr=np.arange(lmax+1)
clttth=((larr+10.)/(l0+10.))**tilt
cleeth=((larr+30.)/(l0+30.))**tilt
#clteth=0.3*np.sin(larr/(30.*((larr+100.)/100)*0.1))*np.sqrt(clttth*cleeth)
clteth=0.3*np.sqrt(clttth*cleeth)
clbbth=np.zeros_like(cleeth)

if use_cont :
    tilt_fg=-2.0
    l0_fg=100.
    clttfg=50*((larr+10.)/(l0_fg+10.))**tilt_fg
    cleefg=50*((larr+30.)/(l0_fg+30.))**tilt_fg
    cltefg=0.9*np.sqrt(clttfg*cleefg)
    clbbfg=cleefg

#Make mask
theta,phi=hp.pix2ang(nside,np.arange(hp.nside2npix(nside))); phi[phi>=np.pi]-=2*np.pi
cth0=-np.sqrt(fsky); cthf= np.sqrt(fsky)
theta0=np.arccos(cthf); thetaf=np.arccos(cth0)
phi0=-np.pi*np.sqrt(fsky); phif=np.pi*np.sqrt(fsky)
ids=np.where((theta>theta0) & (theta<thetaf) &
             (phi>phi0) & (phi<phif))[0]
mask_raw=np.zeros(hp.nside2npix(nside)); mask_raw[ids]=1.
cths=cth0+(cthf-cth0)*np.random.rand(nholes)
phis=phi0+(phif-phi0)*np.random.rand(nholes)
ths=np.arccos(cths)
vs=np.transpose(np.array([np.sin(ths)*np.cos(phis),np.sin(ths)*np.sin(phis),np.cos(ths)]))
for i in np.arange(nholes) :
  v=vs[i]
  mask_raw[hp.query_disc(nside,vs[i],rholes*np.pi/180)]=0
if aposize>0 :
    mask=nmt.mask_apodization(mask_raw,aposize,apotype='C1')
else :
    mask=mask_raw
print np.mean(mask_raw), np.mean(mask)

if use_cont :
    fgt,fgq,fgu=hp.synfast([clttfg,cleefg,clbbfg,cltefg,0*cltefg,0*cltefg],nside,verbose=False,new=True)

#Random fields
def get_field() :
    mppt,mppq,mppu=hp.synfast([clttth,cleeth,clbbth,clteth,0*clteth,0*clteth],nside,verbose=False,new=True)
    if use_cont :
        mppt+=alpha_cont*fgt
        mppq+=alpha_cont*fgq
        mppu+=alpha_cont*fgu
        ff0=nmt.NmtField(mask,[mppt*mask_raw],templates=[[fgt]])
        ff2=nmt.NmtField(mask,[mppq*mask_raw,mppu*mask_raw],[[fgq,fgu]])
    else :
        ff0=nmt.NmtField(mask,[mppt*mask_raw])
        ff2=nmt.NmtField(mask,[mppq*mask_raw,mppu*mask_raw])
    return mppt,mppq,mppu,ff0,ff2
mpt,mpq,mpu,f0,f2=get_field()

'''
hp.mollview(mask);
hp.mollview(mpt*mask_raw)
hp.mollview(mpq*mask_raw)
hp.mollview(mpu*mask_raw)
'''

#Make bins
d_ell=int(1./fsky)
b=nmt.NmtBin(nside,nlb=d_ell)
ell_eff=b.get_effective_ells()

#Workspace
print "Computing coupling matrix"
print "TT"
w00=nmt.NmtWorkspace();
if os.path.isfile(prefix+"_w00.dat") :
    w00.read_from(prefix+"_w00.dat");
else :
    w00.compute_coupling_matrix(f0,f0,b);
    w00.write_to(prefix+"_w00.dat");
print "TE"
w02=nmt.NmtWorkspace();
if os.path.isfile(prefix+"_w02.dat") :
    w02.read_from(prefix+"_w02.dat");
else :
    w02.compute_coupling_matrix(f0,f2,b);
    w02.write_to(prefix+"_w02.dat");
print "EE"
w22=nmt.NmtWorkspace();
if os.path.isfile(prefix+"_w22.dat") :
    w22.read_from(prefix+"_w22.dat");
else :
    w22.compute_coupling_matrix(f2,f2,b);
    w22.write_to(prefix+"_w22.dat");
print "yeah"

#Binned theory
clttth_binned=w00.decouple_cell(w00.couple_cell([clttth]))[0]
clteth_binned,cltbth_binned=w02.decouple_cell(w02.couple_cell([clteth,clbbth]))
cleeth_binned,clebth_binned,clbeth_binned,clbbth_binned=w22.decouple_cell(w22.couple_cell([cleeth,clbbth,
                                                                                           clbbth,clbbth,]))

'''
plt.figure()
plt.plot(larr,clttth,'r-'); plt.plot(ell_eff,clttth_binned,'r--')
plt.plot(larr,np.fabs(clteth),'g-'); plt.plot(ell_eff,np.fabs(clteth_binned),'g--')
plt.plot(larr,cleeth,'b-'); plt.plot(ell_eff,cleeth_binned,'b--')
'''

cls_tt_all=[]; cls_tt_all_nmt=[]
cls_te_all=[]; cls_te_all_nmt=[]
cls_ee_all=[]; cls_ee_all_nmt=[]
for i in np.arange(nsims) :
    print i
    if not ((os.path.isfile(prefix+"_cl_aft_%04d.txt"%i)) or 
            (os.path.isfile(prefix+"_cl_nmt_%04d.txt"%i))) :
        mpt,mpq,mpu,f0,f2=get_field()
    if os.path.isfile(prefix+"_cl_aft_%04d.txt"%i) :
        larr,cltt,clee,clbb,clte=np.loadtxt(prefix+"_cl_aft_%04d.txt"%i,unpack=True)
    else :
        cltt,clee,clbb,clte,cleb,cltb=hp.anafast([mpt,mpq,mpu])
        np.savetxt(prefix+"_cl_aft_%04d.txt"%i,np.transpose([larr,cltt,clee,clbb,clte]))
    cls_tt_all.append(cltt);
    cls_te_all.append(clte);
    cls_ee_all.append(clee)

    if os.path.isfile(prefix+"_cl_nmt_%04d.txt"%i) :
        ell_eff,cltt,clee,clbb,clte=np.loadtxt(prefix+"_cl_nmt_%04d.txt"%i,unpack=True)
    else :
        if use_cont :
            clb00=nmt.deprojection_bias(f0,f0,[clttth])
            clb02=nmt.deprojection_bias(f0,f2,[clteth,0*clteth])
            clb22=nmt.deprojection_bias(f2,f2,[cleeth,0*clteth,0*clteth,clbbth])
        else :
            clb00=None
            clb02=None
            clb22=None
        cltt=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0),cl_bias=clb00)[0]
        clte,cltb=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2),cl_bias=clb02)
        clee,cleb,clbe,clbb=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2),cl_bias=clb22)
        np.savetxt(prefix+"_cl_nmt_%04d.txt"%i,np.transpose([ell_eff,cltt,clee,clbb,clte]))
    cls_tt_all_nmt.append(cltt);
    cls_te_all_nmt.append(clte);
    cls_ee_all_nmt.append(clee)
cls_tt_all    =np.array(cls_tt_all);
cls_te_all    =np.array(cls_te_all);
cls_ee_all    =np.array(cls_ee_all);
cls_tt_all_nmt=np.array(cls_tt_all_nmt); 
cls_te_all_nmt=np.array(cls_te_all_nmt); 
cls_ee_all_nmt=np.array(cls_ee_all_nmt);
cls_tt_mean    =np.mean(cls_tt_all    ,axis=0);
cls_te_mean    =np.mean(cls_te_all    ,axis=0);
cls_ee_mean    =np.mean(cls_ee_all    ,axis=0);
cls_tt_mean_nmt=np.mean(cls_tt_all_nmt,axis=0);
cls_te_mean_nmt=np.mean(cls_te_all_nmt,axis=0);
cls_ee_mean_nmt=np.mean(cls_ee_all_nmt,axis=0);
cls_tt_err    =np.std(cls_tt_all    ,axis=0)/np.sqrt(nsims+0.);
cls_te_err    =np.std(cls_te_all    ,axis=0)/np.sqrt(nsims+0.);
cls_ee_err    =np.std(cls_ee_all    ,axis=0)/np.sqrt(nsims+0.);
cls_tt_err_nmt=np.std(cls_tt_all_nmt,axis=0)/np.sqrt(nsims+0.);
cls_te_err_nmt=np.std(cls_te_all_nmt,axis=0)/np.sqrt(nsims+0.);
cls_ee_err_nmt=np.std(cls_ee_all_nmt,axis=0)/np.sqrt(nsims+0.);

'''
plt.figure()
plt.errorbar(larr   ,cls_tt_mean/clttth,yerr=cls_tt_err/clttth,fmt='r-')
plt.errorbar(ell_eff,cls_tt_mean_nmt/clttth_binned,yerr=cls_tt_err_nmt/clttth_binned,fmt='c-')

plt.figure()
plt.errorbar(larr   ,cls_te_mean/clteth,yerr=cls_te_err/clteth,fmt='r-')
plt.errorbar(ell_eff,cls_te_mean_nmt/clteth_binned,yerr=cls_te_err_nmt/clteth_binned,fmt='c-')

plt.figure()
plt.errorbar(larr   ,cls_ee_mean/cleeth,yerr=cls_ee_err/cleeth,fmt='r-')
plt.errorbar(ell_eff,cls_ee_mean_nmt/cleeth_binned,yerr=cls_ee_err_nmt/cleeth_binned,fmt='c-')

plt.show()
'''
