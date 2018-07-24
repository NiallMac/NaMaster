import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import rc
import matplotlib
import pymaster as nmt
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

nside=256
nsims=1000
prefix_clean="tests_sph/run_pure01_ns%d_cont1"%nside
#prefix_dirty="tests_flat/run_mask1_cont1_apo0.00_no_deproj"

def tickfs(ax,x=True,y=True) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

def read_cls(fname) :
    l,cee,ceb,cbe,cbb=np.loadtxt(fname,unpack=True);
    id_good=np.where(l<2*nside)[0]
    return l[id_good],cee[id_good],ceb[id_good],cbe[id_good],cbb[id_good]
l_th,clEE_th,clEB_th,clBE_th,clBB_th=read_cls(prefix_clean+"_cl_th.txt")
ndof=len(l_th)

print "Reading"
clEE_clean=[]; clEB_clean=[]; clBB_clean=[];
#clEE_dirty=[]; clEB_dirty=[]; clBB_dirty=[];
for i in np.arange(nsims) :
    ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_clean+"_cl_%04d.txt"%(i+1))
    clEE_clean.append(ccee); clEB_clean.append(cceb); clBB_clean.append(ccbb);
    #ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_dirty+"_cl_%04d.txt"%(i+1))
    #clEE_dirty.append(ccee); clEB_dirty.append(cceb); clBB_dirty.append(ccbb);
clEE_clean=np.array(clEE_clean); clEB_clean=np.array(clEB_clean); clBB_clean=np.array(clBB_clean); 
#clEE_dirty=np.array(clEE_dirty); clEB_dirty=np.array(clEB_dirty); clBB_dirty=np.array(clBB_dirty);

print "Computing statistics"
def compute_stats(y,y_th) :
    mean=np.mean(y,axis=0)
    cov=np.mean(y[:,:,None]*y[:,None,:],axis=0)-mean[:,None]*mean[None,:]
    icov=np.linalg.inv(cov)
    chi2_red=np.dot(mean-y_th,np.dot(icov,mean-y_th))*nsims
    chi2_all=np.sum((y-y_th)*np.sum(icov[None,:,:]*(y-y_th)[:,None,:],axis=2),axis=1)

    return mean,cov,icov,chi2_red,chi2_all

clEE_clean_mean,clEE_clean_cov,clEE_clean_icov,clEE_clean_chi2r,clEE_clean_chi2all=compute_stats(clEE_clean,clEE_th)
clEB_clean_mean,clEB_clean_cov,clEB_clean_icov,clEB_clean_chi2r,clEB_clean_chi2all=compute_stats(clEB_clean,clEB_th)
clBB_clean_mean,clBB_clean_cov,clBB_clean_icov,clBB_clean_chi2r,clBB_clean_chi2all=compute_stats(clBB_clean,clBB_th)
#clEE_dirty_mean,clEE_dirty_cov,clEE_dirty_icov,clEE_dirty_chi2r,clEE_dirty_chi2all=compute_stats(clEE_dirty,clEE_th)
#clEB_dirty_mean,clEB_dirty_cov,clEB_dirty_icov,clEB_dirty_chi2r,clEB_dirty_chi2all=compute_stats(clEB_dirty,clEB_th)
#clBB_dirty_mean,clBB_dirty_cov,clBB_dirty_icov,clBB_dirty_chi2r,clBB_dirty_chi2all=compute_stats(clBB_dirty,clBB_th)
m,cov,icov,chi2r,chi2all=compute_stats(np.vstack((clEE_clean.T,clEB_clean.T,clBB_clean.T)).T,
                                       np.vstack((clEE_th,clEB_th,clBB_th)).flatten())
print(chi2r,len(m),1-st.chi2.cdf(chi2r,len(m)))

#Plot covariance
plt.figure();
ax=plt.gca();
im=ax.imshow(cov/np.sqrt(np.diag(cov)[None,:]*np.diag(cov)[:,None]),
             interpolation='nearest',cmap=plt.cm.Greys);
for i in np.arange(2)+1 :
    ax.plot([i*ndof,i*ndof],[0,3*ndof],'k--',lw=1)
    ax.plot([0,3*ndof],[i*ndof,i*ndof],'k--',lw=1)
ax.set_xlim([0,3*ndof])
ax.set_ylim([3*ndof,0])
ax.set_xticks(ndof*(np.arange(3)+0.5))
ax.set_yticks(ndof*(np.arange(3)+0.5))
ax.set_xticklabels(['$EE$','$EB$','$BB$'])
ax.set_yticklabels(['$EE$','$EB$','$BB$'])
tickfs(ax)
plt.colorbar(im)
plt.savefig("plots_paper/val_covar_cmb_sph.pdf",bbox_inches='tight')

#Plot residuals
cols=plt.cm.rainbow(np.linspace(0,1,3))
#plot_dirty=False
fig=plt.figure()
ax=fig.add_axes((0.12,0.3,0.78,0.6))
ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
ax.plot([-1,-1],[-1,-1],'k--',label='${\\rm Input}$')
ic=0
ax.plot(l_th,clEE_clean_mean,label='$EE$',c=cols[ic])
#if plot_dirty :
#    ax.plot(l_th,clEE_dirty_mean,'-.',c=cols[ic]);
ax.plot(l_th,clEE_th,'--',c=cols[ic]);
ic+=1
ax.plot(l_th,clEB_clean_mean,label='$EB$',c=cols[ic]);
#if plot_dirty :
#    ax.plot(l_th,clEB_dirty_mean,'-.',c=cols[ic]);
ic+=1
ax.plot(l_th,clBB_clean_mean,label='$BB$',c=cols[ic]);
ax.plot(l_th,clBB_th,'--',c=cols[ic]);
#if plot_dirty :
#    ax.plot(l_th,clBB_dirty_mean,'-.',c=cols[ic]);
ic+=1
#if plot_dirty : 
#    ax.plot([-1,-1],[-1,-1],'k-.' ,label='${\\rm No\\,\\,deproj.}$')
#if plot_dirty : 
#    ax.set_ylim([4E-14,5E-6])
#    ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=3,labelspacing=0.1)
#else :
ax.set_ylim([2E-8,7E-3])
ax.legend(loc='upper left',frameon=False,fontsize=14,ncol=3,labelspacing=0.1)
ax.set_xlim([0,515])
ax.set_yscale('log');
tickfs(ax)
ax.set_xticks([])
#ax.set_yticks([1E-9,1E-7,1E-5,1E-3])
ax.set_ylabel('$C_\\ell$',fontsize=15)
ax=fig.add_axes((0.12,0.1,0.78,0.2))
ic=0
ax.errorbar(l_th  ,(clEE_clean_mean-clEE_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clEE_clean_cov)),
            yerr=np.ones(ndof),label='$EE$',fmt='.',c=cols[ic]); ic+=1
ax.errorbar(l_th+2,(clEB_clean_mean-clEB_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clEB_clean_cov)),
            yerr=np.ones(ndof),label='$EB$',fmt='.',c=cols[ic]); ic+=1
ax.errorbar(l_th+4,(clBB_clean_mean-clBB_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clBB_clean_cov)),
            yerr=np.ones(ndof),label='$BB$',fmt='.',c=cols[ic]); ic+=1
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\Delta C_\\ell/\\sigma_\\ell$',fontsize=15)
ax.set_ylim([-6,6])
ax.set_xlim([0,515])
ax.set_yticks([-4,0,4])
tickfs(ax)
plt.savefig("plots_paper/val_cl_cmb_sph.pdf",bbox_inches='tight')

#Plot chi2 dist
xr=[ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof)]
x=np.linspace(xr[0],xr[1],256)
pdf=st.chi2.pdf(x,ndof)

plt.figure(figsize=(10,4))
ax=[plt.subplot(1,3,i+1) for i in range(3)]
plt.subplots_adjust(wspace=0, hspace=0)

h,b,p=ax[0].hist(clEE_clean_chi2all,bins=40,density=True,range=xr)
ax[0].text(0.8,0.9,'$EE$',transform=ax[0].transAxes,fontsize=14)
ax[0].plot([clEE_clean_chi2r,clEE_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
ax[0].set_xlabel('$\\chi^2$',fontsize=15)
ax[0].set_ylabel('$P(\\chi^2)$',fontsize=15)
print('EE : %.3lE'%(1-st.chi2.cdf(clEE_clean_chi2r,ndof)))

h,b,p=ax[1].hist(clEB_clean_chi2all,bins=40,density=True,range=xr)
ax[1].text(0.8,0.9,'$EB$',transform=ax[1].transAxes,fontsize=14)
ax[1].plot([clEB_clean_chi2r,clEB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('EB : %.3lE'%(1-st.chi2.cdf(clEB_clean_chi2r,ndof)))

h,b,p=ax[2].hist(clBB_clean_chi2all,bins=40,density=True,range=xr)
ax[2].text(0.8,0.9,'$BB$',transform=ax[2].transAxes,fontsize=14)
ax[2].plot([clBB_clean_chi2r,clBB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('BB : %.3lE'%(1-st.chi2.cdf(clBB_clean_chi2r,ndof)))

for a in ax :
    a.set_xlabel('$\\chi^2$',fontsize=15)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
for a in ax :
    tickfs(a)
    a.set_xlim([ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2.*ndof)])
    a.set_ylim([0,1.4*np.amax(pdf)])
    a.plot(x,pdf,'k-',label='$P(\\chi^2)$')
    a.plot([ndof,ndof],[0,1.4*np.amax(pdf)],'k--',label='$N_{\\rm dof}$')
    a.plot([-1,-1],[-1,-1],'k-.',label='$\\chi^2_{\\rm mean}$')
ax[0].legend(loc='upper left',fontsize=12,frameon=False)
plt.savefig("plots_paper/val_chi2_cmb_sph.pdf",bbox_inches='tight')

print "Computing bandpower weights"
ls=np.arange(3*nside,dtype=int)
bpws=np.zeros(3*nside,dtype=int)-1
weights=np.ones(3*nside)
bpw_edges=[2,9,17]
while bpw_edges[-1]<3*nside :
    bpw_edges.append(min(bpw_edges[-1]+12,3*nside))
bpw_edges=np.array(bpw_edges)
for ib,b0 in enumerate(bpw_edges[:-1]) :
    bpws[b0:bpw_edges[ib+1]]=ib
    weights[b0:bpw_edges[ib+1]]=1./(bpw_edges[ib+1]-b0+0.)
b=nmt.NmtBin(nside,ells=ls,bpws=bpws,weights=weights)
w=nmt.NmtWorkspace();
w.read_from(prefix_clean+"_w22.dat")
nbpw=b.get_n_bands()
wmat=np.zeros([nbpw,3*nside])
iden=np.diag(np.ones(4*3*nside))
for l in range(3*nside) :
    if l%100==0 :
        print(l,3*nside)
    wmat[:,l]=w.decouple_cell(w.couple_cell(iden[3*3*nside+l].reshape([4,3*nside])))[3]
plt.figure();
ax=plt.gca()
for ib in np.arange(nbpw) :
    ax.plot(np.arange(3*nside),wmat[ib],'k-',lw=1)
    wbin=np.zeros(3*nside); wbin[b.get_ell_list(ib)]=b.get_weight_list(ib)
    ax.plot(np.arange(3*nside),wbin,'r-',lw=1)
ax.plot([-1,-1],[-1,-1],'k-',lw=1,label='${\\rm Exact}$')
ax.plot([-1,-1],[-1,-1],'r-',lw=1,label='${\\rm Binning\\,\\,approximation}$')
ax.set_xlim([100,200])
ax.set_ylim([-0.03,0.11])
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('${\\rm Bandpower\\,\\,windows}$',fontsize=15)
ax.legend(loc='lower left',ncol=2,frameon=False,fontsize=14)
tickfs(ax)
plt.savefig("plots_paper/val_cmb_bpw.pdf",bbox_inches='tight')
plt.show()

