from __future__ import print_function
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import os
import sys
import data.flatmaps as fm


DTOR=np.pi/180

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--wo-mask', dest='wo_mask', default=False, action='store_true',
                  help='Set if you don\'t want to use a mask')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--aposize', dest='aposize', default=0.0, type=float,
                  help='Mask apodization (in degrees)')
(o, args) = parser.parse_args()

nsims=o.isim_end-o.isim_ini+1
w_mask=not o.wo_mask
w_cont=not o.wo_cont

#Switch off contaminants and inhomogeneous noiseif there's no mask
if not w_mask :
    w_cont=False
    w_nvar=False

#Create output directory
predir="tests_flat"
os.system("mkdir -p "+predir)
prefix=predir+"/run_mask%d_cont%d_apo%.2lf"%(w_mask,w_cont,o.aposize)
fname_mask=prefix+"_mask"

#Read theory power spectra
l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("data/cls_lss.txt",unpack=True)    
cltt[0]=0; clee[0]=0; clbb[0]=0; clte[0]=0;
nltt[0]=0; nlee[0]=0; nlbb[0]=0; nlte[0]=0;
if o.plot_stuff :
    plt.figure()
    plt.plot(l,cltt,'r-',label='$\\delta_g\\times\\delta_g$')
    plt.plot(l,clte,'b-',label='$\\delta_g\\times\\gamma_g$')
    plt.plot(l,clee,'y-',label='$\\gamma_g\\times\\gamma_g$')
    plt.plot(l,nltt,'r--',label='$N_\\delta$')
    plt.plot(l,nlee,'y--',label='$N_\\gamma$')
    plt.loglog(); plt.legend(ncol=2)
    plt.xlabel('$\\ell$',fontsize=16);
    plt.ylabel('$C_\\ell$',fontsize=16);
    plt.savefig(prefix+'_celltheory.png',bbox_inches='tight')
    plt.savefig(prefix+'_celltheory.pdf',bbox_inches='tight')

#Initialize pixelization from mask
fmi,mask_hsc=fm.read_flat_map("data/mask_lss_flat.fits")
if not os.path.isfile(fname_mask+'.fits') :
    if w_mask :
        mask_raw=mask_hsc.copy()
        if o.aposize>0 :
            mask=nmt.mask_apodization_flat(mask_raw.reshape([fmi.ny,fmi.nx]),
                                           fmi.lx_rad,fmi.ly_rad,o.aposize).flatten()
        else :
            mask=mask_raw.copy()
    else :
        mask=np.ones_like(mask_hsc)

    fmi.write_flat_map(fname_mask+".fits",mask)
fdum,mask=fm.read_flat_map(fname_mask+".fits")
if o.plot_stuff :
    fmi.view_map(mask,title='Mask')
    plt.savefig(prefix+'_mask.png',bbox_inches='tight')
    plt.savefig(prefix+'_mask.pdf',bbox_inches='tight')
fsky=fmi.lx_rad*fmi.ly_rad*np.sum(mask)/(4*np.pi*fmi.nx*fmi.ny)

#Read contaminant maps
if w_cont :
    fgt=np.zeros([2,1,len(mask)])
    dum,fgt[0,0,:]=fm.read_flat_map("data/cont_lss_star_flat.fits") #Stars
    dum,fgt[1,0,:]=fm.read_flat_map("data/cont_lss_dust_flat.fits") #Dust
    fgp=np.zeros([2,2,len(mask)])
    dum,[fgp[0,0,:],fgp[0,1,:]]=fm.read_flat_map("data/cont_wl_psf_flat.fits",i_map=-1) #PSF
    dum,[fgp[1,0,:],fgp[1,1,:]]=fm.read_flat_map("data/cont_wl_ss_flat.fits",i_map=-1) #Small-scales
    if o.plot_stuff :
        fmi.view_map(np.sum(fgt,axis=0)[0,:]*mask,title='Total LSS contaminant')
        fmi.view_map(np.sum(fgp,axis=0)[0,:]*mask,title='Total WL contaminant (Q)')
        fmi.view_map(np.sum(fgp,axis=0)[1,:]*mask,title='Total WL contaminant (U)')

#Binning scheme
ell_min=max(2*np.pi/fmi.lx_rad,2*np.pi/fmi.ly_rad)
ell_max=min(fmi.nx*np.pi/fmi.lx_rad,fmi.ny*np.pi/fmi.ly_rad)
d_ell=2*ell_min
n_ell=int((ell_max-ell_min)/d_ell)-1
l_bpw=np.zeros([2,n_ell])
l_bpw[0,:]=ell_min+np.arange(n_ell)*d_ell
l_bpw[1,:]=l_bpw[0,:]+d_ell
b=nmt.NmtBinFlat(l_bpw[0,:],l_bpw[1,:])

#Generate some initial fields
print(" - Res(x): %.3lf arcmin. Res(y): %.3lf arcmin."%(fmi.lx*60/fmi.nx,fmi.ly*60/fmi.ny))
print(" - lmax = %d, lmin = %d"%(int(ell_max),int(ell_min)))
def get_fields() :
    st,sq,su=nmt.synfast_flat(int(fmi.nx),int(fmi.ny),fmi.lx_rad,fmi.ly_rad,
                              [cltt+nltt,clee+nlee,clbb+nlbb,clte+nlte],pol=True)
    st=st.flatten(); sq=sq.flatten(); su=su.flatten()
    if w_cont :
        st+=np.sum(fgt,axis=0)[0,:]; sq+=np.sum(fgp,axis=0)[0,:]; su+=np.sum(fgp,axis=0)[1,:];
        ff0=nmt.NmtFieldFlat(fmi.lx_rad,fmi.ly_rad,mask.reshape([fmi.ny,fmi.nx]),
                             [st.reshape([fmi.ny,fmi.nx])],
                             templates=fgt.reshape([2,1,fmi.ny,fmi.nx]))
        ff2=nmt.NmtFieldFlat(fmi.lx_rad,fmi.ly_rad,mask.reshape([fmi.ny,fmi.nx]),
                             [sq.reshape([fmi.ny,fmi.nx]),su.reshape([fmi.ny,fmi.nx])],
                             templates=fgp.reshape([2,2,fmi.ny,fmi.nx]))
    else :
        ff0=nmt.NmtFieldFlat(fmi.lx_rad,fmi.ly_rad,mask.reshape([fmi.ny,fmi.nx]),
                             [st.reshape([fmi.ny,fmi.nx])])
        ff2=nmt.NmtFieldFlat(fmi.lx_rad,fmi.ly_rad,mask.reshape([fmi.ny,fmi.nx]),
                             [sq.reshape([fmi.ny,fmi.nx]),su.reshape([fmi.ny,fmi.nx])])
    return ff0,ff2

np.random.seed(1000)
f0,f2=get_fields()

if o.plot_stuff :
    fmi.view_map(f0.get_maps()[0].flatten()*mask,title='$\\delta$')
    fmi.view_map(f2.get_maps()[0].flatten()*mask,title='$\\gamma_1$')
    fmi.view_map(f2.get_maps()[1].flatten()*mask,title='$\\gamma_2$')

#Use initial fields to generate coupling matrix
w00=nmt.NmtWorkspaceFlat();
if not os.path.isfile(prefix+"_w00.dat") :
    print("Computing 00")
    w00.compute_coupling_matrix(f0,f0,b)
    w00.write_to(prefix+"_w00.dat");
else :
    w00.read_from(prefix+"_w00.dat")
w02=nmt.NmtWorkspaceFlat();
if not os.path.isfile(prefix+"_w02.dat") :
    print("Computing 02")
    w02.compute_coupling_matrix(f0,f2,b)
    w02.write_to(prefix+"_w02.dat");
else :
    w02.read_from(prefix+"_w02.dat")
w22=nmt.NmtWorkspaceFlat();
if not os.path.isfile(prefix+"_w22.dat") :
    print("Computing 22")
    w22.compute_coupling_matrix(f2,f2,b)
    w22.write_to(prefix+"_w22.dat");
else :
    w22.read_from(prefix+"_w22.dat")
    
#Generate theory prediction
if not os.path.isfile(prefix+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w00.decouple_cell(w00.couple_cell(l,np.array([cltt])))
    cl02_th=w02.decouple_cell(w02.couple_cell(l,np.array([clte,0*clte])))
    cl22_th=w22.decouple_cell(w22.couple_cell(l,np.array([clee,0*clee,0*clbb,clbb])))
    np.savetxt(prefix+"_cl_th.txt",
               np.transpose([b.get_effective_ells(),cl00_th[0],cl02_th[0],cl02_th[1],
                             cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))
else :
    cl00_th=np.zeros([1,b.get_n_bands()])
    cl02_th=np.zeros([2,b.get_n_bands()])
    cl22_th=np.zeros([4,b.get_n_bands()])
    dum,cl00_th[0],cl02_th[0],cl02_th[1],cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]=np.loadtxt(prefix+"_cl_th.txt",unpack=True)
    
#Compute noise and deprojection bias
if not os.path.isfile(prefix+"_clb00.npy") :
    print("Computing deprojection and noise bias 00")
    #Compute noise bias
    clb00=w00.couple_cell(l,np.array([nltt]))
    #Compute deprojection bias
    if w_cont :
        clb00+=nmt.deprojection_bias_flat(f0,f0,b,l,[cltt])
    np.save(prefix+"_clb00",clb00)
else :
    clb00=np.load(prefix+"_clb00.npy")
if not os.path.isfile(prefix+"_clb02.npy") :
    print("Computing deprojection and noise bias 02")
    clb02=w02.couple_cell(l,np.array([nlte,0*nlte]))
    if w_cont :
        clb02+=nmt.deprojection_bias_flat(f0,f2,b,l,[clte,0*clte])
    np.save(prefix+"_clb02",clb02)
else :
    clb02=np.load(prefix+"_clb02.npy")
if not os.path.isfile(prefix+"_clb22.npy") :
    print("Computing deprojection and noise bias 22")
    clb22=w22.couple_cell(l,np.array([nlee,0*nlee,0*nlbb,nlbb]))
    if w_cont :
        clb22+=nmt.deprojection_bias_flat(f2,f2,b,l,[clee,0*clee,0*clbb,clbb])
    np.save(prefix+"_clb22",clb22)
else :
    clb22=np.load(prefix+"_clb22.npy")

#Compute mean and variance over nsims simulations
cl00_all=[]
cl02_all=[]
cl22_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix+"_cl_%04d.txt"%(o.isim_ini+i)) :
        f0,f2=get_fields()
        cl00=w00.decouple_cell(nmt.compute_coupled_cell_flat(f0,f0,b),cl_bias=clb00)
        cl02=w02.decouple_cell(nmt.compute_coupled_cell_flat(f0,f2,b),cl_bias=clb02)
        cl22=w22.decouple_cell(nmt.compute_coupled_cell_flat(f2,f2,b),cl_bias=clb22)
        np.savetxt(prefix+"_cl_%04d.txt"%(o.isim_ini+i),
                   np.transpose([b.get_effective_ells(),cl00[0],cl02[0],cl02[1],
                                 cl22[0],cl22[1],cl22[2],cl22[3]]))
    cld=np.loadtxt(prefix+"_cl_%04d.txt"%(o.isim_ini+i),unpack=True)
    cl00_all.append([cld[1]])
    cl02_all.append([cld[2],cld[3]])
    cl22_all.append([cld[4],cld[5],cld[6],cld[7]])
cl00_all=np.array(cl00_all)
cl02_all=np.array(cl02_all)
cl22_all=np.array(cl22_all)

#Plot results
if o.plot_stuff :
    l_eff=b.get_effective_ells()
    cols=plt.cm.rainbow(np.linspace(0,1,6))
    plt.figure()
    plt.errorbar(l_eff+4,np.mean(cl00_all,axis=0)[0]/cl00_th[0]-1,
                 yerr=np.std(cl00_all,axis=0)[0]/cl00_th[0]/np.sqrt(nsims+0.),
                 label='$\\delta\\times\\delta$',fmt='ro')
    plt.errorbar(l_eff+4,np.mean(cl02_all,axis=0)[0]/cl02_th[0]-1,
                 yerr=np.std(cl02_all,axis=0)[0]/cl02_th[0]/np.sqrt(nsims+0.),
                 label='$\\delta\\times\\gamma_E$',fmt='go')
    plt.errorbar(l_eff+8,np.mean(cl22_all,axis=0)[0]/cl22_th[0]-1,
                 yerr=np.std(cl22_all,axis=0)[0]/cl22_th[0]/np.sqrt(nsims+0.),
                 label='$\\gamma_E\\times\\gamma_E$',fmt='bo')
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$\\Delta C_\\ell/C_\\ell$',fontsize=16)
    plt.legend(loc='lower left',frameon=False,fontsize=16)
    plt.xscale('log')
    plt.savefig(prefix+'_celldiff.png',bbox_inches='tight')
    plt.savefig(prefix+'_celldiff.pdf',bbox_inches='tight')


    import scipy.stats as st
    chi2_00=np.sum(((cl00_all[:,:,:]-cl00_th[None,:,:])/np.std(cl00_all,axis=0)[None,:,:])**2,axis=2)
    chi2_02=np.sum(((cl02_all[:,:,:]-cl02_th[None,:,:])/np.std(cl02_all,axis=0)[None,:,:])**2,axis=2)
    chi2_22=np.sum(((cl22_all[:,:,:]-cl22_th[None,:,:])/np.std(cl22_all,axis=0)[None,:,:])**2,axis=2)
    nsim,nel00,ndof=cl00_all.shape
    nsim,nel02,ndof=cl02_all.shape
    nsim,nel22,ndof=cl22_all.shape

    x=np.linspace(ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof),256)
    pdf=st.chi2.pdf(x,ndof)

    
    plt.figure(figsize=(10,7))
    ax=[plt.subplot(2,3,i+1) for i in range(6)]
    plt.subplots_adjust(wspace=0, hspace=0)
    
    h,b,p=ax[0].hist(chi2_00[:,0],bins=40,density=True)
    ax[0].text(0.75,0.9,'$\\delta\\times\\delta$'    ,transform=ax[0].transAxes)
    ax[0].set_ylabel('$P(\\chi^2)$')

    h,b,p=ax[1].hist(chi2_02[:,0],bins=40,density=True)
    ax[1].text(0.75,0.9,'$\\delta\\times\\gamma_E$'  ,transform=ax[1].transAxes)

    h,b,p=ax[2].hist(chi2_02[:,1],bins=40,density=True)
    ax[2].text(0.75,0.9,'$\\delta\\times\\gamma_B$'  ,transform=ax[2].transAxes)

    h,b,p=ax[3].hist(chi2_22[:,0],bins=40,density=True)
    ax[3].text(0.75,0.9,'$\\gamma_E\\times\\gamma_E$',transform=ax[3].transAxes)
    ax[3].set_xlabel('$\\chi^2$')
    ax[3].set_ylabel('$P(\\chi^2)$')

    h,b,p=ax[4].hist(chi2_22[:,1],bins=40,density=True)
    ax[4].text(0.75,0.9,'$\\gamma_E\\times\\gamma_B$',transform=ax[4].transAxes)

    h,b,p=ax[5].hist(chi2_22[:,3],bins=40,density=True)
    ax[5].text(0.75,0.9,'$\\gamma_B\\times\\gamma_B$',transform=ax[5].transAxes)

    for a in ax[:3] :
        a.set_xticklabels([])
    for a in ax[3:] :
        a.set_xlabel('$\\chi^2$')
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])
    for a in ax :
        a.set_xlim([ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2.*ndof)])
        a.set_ylim([0,1.4*np.amax(pdf)])
        a.plot([ndof,ndof],[0,1.4*np.amax(pdf)],'k--',label='$N_{\\rm dof}$')
        a.plot(x,pdf,'k-',label='$P(\\chi^2,N_{\\rm dof})$')
    ax[3].legend(loc='upper left',frameon=False)
    plt.savefig(prefix+'_distributions.png',bbox_inches='tight')
    plt.savefig(prefix+'_distributions.pdf',bbox_inches='tight')
    
    ic=0
    plt.figure()
    plt.plot(l_eff,np.mean(cl00_all,axis=0)[0],
             label='$\\delta\\times\\delta$',c=cols[ic])
    plt.plot(l_eff,cl00_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl02_all,axis=0)[0],
             label='$\\delta\\times\\gamma_E$',c=cols[ic]);
    plt.plot(l_eff,cl02_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl02_all,axis=0)[1],
             label='$\\delta\\times\\gamma_B$',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[0],
             label='$\\gamma\\times\\gamma_E$',c=cols[ic]);
    plt.plot(l_eff,cl22_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[1],
             label='$\\gamma_E\\times\\gamma_B$',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[3],
             label='$\\gamma_B\\times\\gamma_B$',c=cols[ic]); ic+=1
    plt.loglog()
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C_\\ell$',fontsize=16)
    plt.legend(loc='lower left',frameon=False,fontsize=14,ncol=2)
    plt.savefig(prefix+'_cellfull.png',bbox_inches='tight')
    plt.savefig(prefix+'_cellfull.pdf',bbox_inches='tight')
    plt.show()
