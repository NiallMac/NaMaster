import nmtlib as lib
import numpy as np

class NmtWorkspace(object) :
    """
    NmtWorkspace objects are used to compute and store the coupling matrix associated with an incomplete sky coverage, and used in the MASTER algorithm. When initialized, this object is practically empty. The information describing the coupling matrix must be computed or read from a file afterwards.
    """
    def __init__(self) :
        self.wsp=None

    def __del__(self) :
        if(self.wsp!=None) :
            lib.workspace_free(self.wsp)

    def read_from(self,fname) :
        """
        Reads the contents of an NmtWorkspace object from a file (encoded using an internal binary format).

        :param str fname: input file name
        """
        if self.wsp!=None :
            lib.workspace_free(self.wsp)
        self.wsp=lib.workspace_read(fname);
        
    def compute_coupling_matrix(self,fl1,fl2,bins) :
        """
        Computes coupling matrix associated with the cross-power spectrum of two NmtFields and an NmtBin binning scheme.

        :param NmtField fl1,fl2: fields to correlate
        :param NmtBin bin: binning scheme
        """
        if self.wsp!=None :
            lib.workspace_free(self.wsp)
        self.wsp=lib.compute_coupling_matrix(fl1.fl,fl2.fl,bins.bin)

    def write_to(self,fname) :
        """
        Writes the contents of an NmtWorkspace object to a file (encoded using an internal binary format).

        :param str fname: output file name
        """
        if self.wsp==None :
            raise KeyError("Must initialize workspace before writing")
        lib.workspace_write(self.wsp,fname)

    def couple_cell(self,cl_in) :
        """
        Convolves a set of input power spectra with a coupling matrix (see Eq. 6 of the C API documentation).

        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspace object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field and 4 for two spin-2 fields).
        :return: coupled power spectrum
        """
        if((len(cl_in)!=self.wsp.ncls) or (len(cl_in[0])!=self.wsp.lmax+1)) :
            raise KeyError("Input power spectrum has wrong shape")
        cl1d=lib.couple_cell_py(self.wsp,cl_in,self.wsp.ncls*(self.wsp.lmax+1))
        clout=np.reshape(cl1d,[self.wsp.ncls,self.wsp.lmax+1])
        return clout

    def decouple_cell(self,cl_in,cl_bias=None,cl_noise=None) :
        """
        Decouples a set of pseudo-Cl power spectra into a set of bandpowers by inverting the binned coupling matrix (se Eq. 4 of the C API documentation).

        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspace object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field and 4 for two spin-2 fields).
        :param cl_bias: bias to the power spectrum associated to contaminant residuals (optional). This can be computed through :func:`pymaster.deprojection_bias`.
        :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations).
        :return: set of decoupled bandpowers
        """
        if((len(cl_in)!=self.wsp.ncls) or (len(cl_in[0])!=self.wsp.lmax+1)) :
            raise KeyError("Input power spectrum has wrong shape")
        if cl_bias!=None :
            if((len(cl_bias)!=self.wsp.ncls) or (len(cl_bias[0])!=self.wsp.lmax+1)) :
                raise KeyError("Input bias power spectrum has wrong shape")
            clb=cl_bias.copy()
        else :
            clb=np.zeros_like(cl_in)
        if cl_noise!=None :
            if((len(cl_noise)!=self.wsp.ncls) or (len(cl_noise[0])!=self.wsp.lmax+1)) :
                raise KeyError("Input noise power spectrum has wrong shape")
            cln=cl_noise.copy()
        else :
            cln=np.zeros_like(cl_in)

        cl1d=lib.decouple_cell_py(self.wsp,cl_in,cln,clb,self.wsp.ncls*self.wsp.bin.n_bands)
        clout=np.reshape(cl1d,[self.wsp.ncls,self.wsp.bin.n_bands])

        return clout

class NmtWorkspaceFlat(object) :
    """
    NmtWorkspaceFlat objects are used to compute and store the coupling matrix associated with an incomplete sky coverage, and used in the flat-sky version of the MASTER algorithm. When initialized, this object is practically empty. The information describing the coupling matrix must be computed or read from a file afterwards.
    """
    def __init__(self) :
        self.wsp=None

    def __del__(self) :
        if(self.wsp!=None) :
            lib.workspace_flat_free(self.wsp)

    def read_from(self,fname) :
        """
        Reads the contents of an NmtWorkspaceFlat object from a file (encoded using an internal binary format).

        :param str fname: input file name
        """
        if self.wsp!=None :
            lib.workspace_flat_free(self.wsp)
        self.wsp=lib.workspace_flat_read(fname);
        
    def compute_coupling_matrix(self,fl1,fl2,bins,nell_rebin=2,method=203,ell_cut_x=[1.,-1.],ell_cut_y=[1.,-1.]) :
        """
        Computes coupling matrix associated with the cross-power spectrum of two NmtFieldFlats and an NmtBinFlat binning scheme.

        :param NmtFieldFlat fl1,fl2: fields to correlate
        :param NmtBinFlat bin: binning scheme
        :param int nell_rebin: number of sub-intervals into which the base k-intervals will be sub-sampled to compute the coupling matrix
        :param int method: algorithm to compute the coupling matrix (only 203 has been fully validated so far).
        :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
        :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
        """
        if self.wsp!=None :
            lib.workspace_flat_free(self.wsp)

        self.wsp=lib.compute_coupling_matrix_flat(fl1.fl,fl2.fl,bins.bin,nell_rebin,method,
                                                  ell_cut_x[0],ell_cut_x[1],ell_cut_y[0],ell_cut_y[1])

    def write_to(self,fname) :
        """
        Writes the contents of an NmtWorkspaceFlat object to a file (encoded using an internal binary format).

        :param str fname: output file name
        """
        if self.wsp==None :
            raise KeyError("Must initialize workspace before writing")
        lib.workspace_flat_write(self.wsp,fname)

    def get_ell_sampling(self) :
        """
        Returns the multipoles at which the coupling matrix has been computed

        :return: list of multipoles
        """
        ells=lib.get_ell_sampling_flat_wsp(self.wsp,self.wsp.nells)

        return ells

    def couple_cell(self,ells,cl_in) :
        """
        Convolves a set of input power spectra with a coupling matrix (see Eq. 6 of the C API documentation).

        :param ells: list of multipoles on which the input power spectra are defined
        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspaceFlat object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field and 4 for two spin-2 fields).
        :return: coupled power spectrum. The coupled power spectra are returned at the multipoles returned by calling :func:`get_ell_sampling` for any of the fields that were used to generate the workspace.
        """
        if((len(cl_in)!=self.wsp.ncls) or (len(cl_in[0])!=len(ells))) :
            raise KeyError("Input power spectrum has wrong shape")
        cl1d=lib.couple_cell_py_flat(self.wsp,ells,cl_in,self.wsp.ncls*self.wsp.fs.n_ell)
        clout=np.reshape(cl1d,[self.wsp.ncls,self.wsp.fs.n_ell])
        return clout

    def decouple_cell(self,cl_in,cl_bias=None,cl_noise=None) :
        """
        Decouples a set of pseudo-Cl power spectra into a set of bandpowers by inverting the binned coupling matrix (se Eq. 4 of the C API documentation).

        :param cl_in: set of input power spectra. The number of power spectra must correspond to the spins of the two fields that this NmtWorkspaceFlat object was initialized with (i.e. 1 for two spin-0 fields, 2 for one spin-0 and one spin-2 field and 4 for two spin-2 fields). These power spectra must be defined at the multipoles returned by :func:`get_ell_sampling` for any of the fields used to create the workspace.
        :param cl_bias: bias to the power spectrum associated to contaminant residuals (optional). This can be computed through :func:`pymaster.deprojection_bias_flat`.
        :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations).
        :return: set of decoupled bandpowers
        """
        if((len(cl_in)!=self.wsp.ncls) or (len(cl_in[0])!=self.wsp.fs.n_ell)) :
            raise KeyError("Input power spectrum has wrong shape")
        if cl_bias!=None :
            if((len(cl_bias)!=self.wsp.ncls) or (len(cl_bias[0])!=self.wsp.fs.n_ell)) :
                raise KeyError("Input bias power spectrum has wrong shape")
            clb=cl_bias.copy()
        else :
            clb=np.zeros_like(cl_in)
        if cl_noise!=None :
            if((len(cl_noise)!=self.wsp.ncls) or (len(cl_noise[0])!=self.wsp.fs.n_ell)) :
                raise KeyError("Input noise power spectrum has wrong shape")
            cln=cl_noise.copy()
        else :
            cln=np.zeros_like(cl_in)

        cl1d=lib.decouple_cell_py_flat(self.wsp,cl_in,cln,clb,self.wsp.ncls*self.wsp.bin.n_bands)
        clout=np.reshape(cl1d,[self.wsp.ncls,self.wsp.bin.n_bands])

        return clout

def gaussian_covariance(wa,wb,cla1b1,cla1b2,cla2b1,cla2b2) :
    """
    Computes Gaussian covariance matrix for power spectra computed with workspaces wa and wb.
    The description above assumes that wa was used to compute the cross correlation of two fields labelled a1 and a2
    (and b1, b2 for wb).
    Note that all fields should have the same resolution, and the predicted input power spectra should be defined
    for all ells <=3*nside (where nside is the HEALPix resolution parameter).

    :param NmtWorkspace wa,wb: workspaces used to compute pseudo-Cl estimators of power spectra.
    :param cla1b1: prediction for the cross-power spectrum between a1 and b1.
    :param cla1b2: prediction for the cross-power spectrum between a1 and b2.
    :param cla2b1: prediction for the cross-power spectrum between a2 and b1.
    :param cla2b2: prediction for the cross-power spectrum between a2 and b2.
    """
    ns=wa.wsp.nside;
    if(wa.wsp.nside!=wb.wsp.nside) :
        raise ValueError("Everything should have the same resolution!")
    if((wa.wsp.ncls!=1) or (wb.wsp.ncls!=1)) :
        raise ValueError("Gaussian covariances only supported for spin-0 fields")
    if((len(cla1b1)!=wa.wsp.lmax+1) or (len(cla1b2)!=wa.wsp.lmax+1) or (len(cla2b1)!=wa.wsp.lmax+1) or (len(cla2b2)!=wa.wsp.lmax+1)) :
        raise ValueError("Input C_ls have a weird length")
    len_a=wa.wsp.ncls*wa.wsp.bin.n_bands
    len_b=wb.wsp.ncls*wb.wsp.bin.n_bands

    covar1d=lib.comp_gaussian_covariance(wa.wsp,wb.wsp,cla1b1,cla1b2,cla2b1,cla2b2,len_a*len_b)
    covar=np.reshape(covar1d,[len_a,len_b])
    return covar

def deprojection_bias(f1,f2,cls_guess) :
    """
    Computes the bias associated to contaminant removal to the cross-pseudo-Cl of two fields.

    :param NmtField f1,f2: fields to correlate
    :param cls_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2.
    :return: deprojection bias power spectra.
    """
    if(len(cls_guess)!=f1.fl.nmaps*f2.fl.nmaps) :
        raise KeyError("Proposal Cell doesn't match number of maps")
    if(len(cls_guess[0])!=f1.fl.lmax+1) :
        raise KeyError("Proposal Cell doesn't match map resolution")
    cl1d=lib.comp_deproj_bias(f1.fl,f2.fl,cls_guess,len(cls_guess)*len(cls_guess[0]))
    cl2d=np.reshape(cl1d,[len(cls_guess),len(cls_guess[0])])

    return cl2d

def deprojection_bias_flat(f1,f2,ells,cls_guess,ell_cut_x=[1.,-1.],ell_cut_y=[1.,-1.]) :
    """
    Computes the bias associated to contaminant removal to the cross-pseudo-Cl of two flat-sky fields. The returned power spectrum is defined at the multipoles returned by the method :func:`get_ell_sampling` of either f1 or f2.

    :param NmtFieldFlat f1,f2: fields to correlate
    :param ells: list of multipoles on which the proposal power spectra are defined
    :param cls_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2.
    :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
    :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
    :return: deprojection bias power spectra.
    """
    if(len(cls_guess)!=f1.fl.nmaps*f2.fl.nmaps) :
        raise KeyError("Proposal Cell doesn't match number of maps")
    if(len(cls_guess[0])!=len(ells)) :
        raise KeyError("Proposal Cell doesn't match map resolution")
    cl1d=lib.comp_deproj_bias_flat(f1.fl,f2.fl,ell_cut_x[0],ell_cut_x[1],ell_cut_y[0],ell_cut_y[1],
                                   ells,cls_guess,len(cls_guess)*len(cls_guess[0]))
    cl2d=np.reshape(cl1d,[len(cls_guess),len(cls_guess[0])])

    return cl2d

def compute_coupled_cell(f1,f2,n_iter=3) :
    """
    Computes the full-sky angular power spectra of two masked fields (f1 and f2) without aiming to deconvolve the mode-coupling matrix. Effectively, this is equivalent to calling the usual HEALPix anafast routine on the masked and contaminant-cleaned maps.

    :param NmtField f1,f2: fields to correlate
    :param int n_iter: number of iterations for SHTs (optional)
    :return: array of coupled power spectra
    """
    if(f1.fl.nside!=f2.fl.nside) :
        raise KeyError("Fields must have same resolution")
    
    cl1d=lib.comp_pspec_coupled(f1.fl,f2.fl,f1.fl.nmaps*f2.fl.nmaps*(f1.fl.lmax+1),n_iter)
    clout=np.reshape(cl1d,[f1.fl.nmaps*f2.fl.nmaps,f1.fl.lmax+1])

    return clout

def compute_coupled_cell_flat(f1,f2,ell_cut_x=[1.,-1.],ell_cut_y=[1.,-1.]) :
    """
    Computes the angular power spectra of two masked flat-sky fields (f1 and f2) without aiming to deconvolve the mode-coupling matrix. Effectively, this is equivalent to computing the map FFTs and averaging over rings of wavenumber.  The returned power spectrum is defined at the multipoles returned by the method :func:`get_ell_sampling` of either f1 or f2.

    :param NmtFieldFlat f1,f2: fields to correlate
    :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
    :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
    :return: array of coupled power spectra
    """
    if((f1.nx!=f2.nx) or (f1.ny!=f2.ny)) :
        raise KeyError("Fields must have same resolution")
    
    cl1d=lib.comp_pspec_coupled_flat(f1.fl,f2.fl,f1.fl.nmaps*f2.fl.nmaps*f1.fl.fs.n_ell,
                                     ell_cut_x[0],ell_cut_x[1],ell_cut_y[0],ell_cut_y[1])
    clout=np.reshape(cl1d,[f1.fl.nmaps*f2.fl.nmaps,f1.fl.fs.n_ell])

    return clout

def compute_full_master(f1,f2,b,cl_noise=None,cl_guess=None,workspace=None) :
    """
    Computes the full MASTER estimate of the power spectrum of two fields (f1 and f2). This is equivalent to successively calling:

    - :func:`pymaster.NmtWorkspace.compute_coupling_matrix`
    - :func:`pymaster.deprojection_bias`
    - :func:`pymaster.compute_coupled_cell`
    - :func:`pymaster.NmtWorkspace.decouple_cell`

    :param NmtField f1,f2: fields to correlate
    :param NmtBin b: binning scheme defining output bandpower
    :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations) (optional).
    :param cl_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2. Needed only to compute the contaminant cleaning bias (optional).
    :param NmtWorkspace workspace: object containing the mode-coupling matrix associated with an incomplete sky coverage. If provided, the function will skip the computation of the mode-coupling matrix and use the information encoded in this object.
    :return: set of decoupled bandpowers
    """
    if(f1.fl.nside!=f2.fl.nside) :
        raise KeyError("Fields must have same resolution")
    if cl_noise!=None :
        if(len(cl_noise)!=f1.fl.nmaps*f2.fl.nmaps) :
            raise KeyError("Wrong length for noise power spectrum")
        cln=cl_noise.copy()
    else :
        cln=np.zeros([f1.fl.nmaps*f2.fl.nmaps,3*f1.fl.nside])
    if cl_guess!=None :
        if(len(cl_guess)!=f1.fl.nmaps*f2.fl.nmaps) :
            raise KeyError("Wrong length for guess power spectrum")
        clg=cl_guess.copy()
    else :
        clg=np.zeros([f1.fl.nmaps*f2.fl.nmaps,3*f1.fl.nside])

    if workspace==None :
        cl1d=lib.comp_pspec(f1.fl,f2.fl,b.bin,None,cln,clg,len(cln)*b.bin.n_bands)
    else :
        cl1d=lib.comp_pspec(f1.fl,f2.fl,b.bin,workspace.wsp,cln,clg,len(cln)*b.bin.n_bands)

    clout=np.reshape(cl1d,[len(cln),b.bin.n_bands])

    return clout

def compute_full_master_flat(f1,f2,b,cl_noise=None,cl_guess=None,ells_guess=None,workspace=None,nell_rebin=2,
                             ell_cut_x=[1.,-1.],ell_cut_y=[1.,-1.]) :
    """
    Computes the full MASTER estimate of the power spectrum of two flat-sky fields (f1 and f2). This is equivalent to successively calling:

    - :func:`pymaster.NmtWorkspaceFlat.compute_coupling_matrix`
    - :func:`pymaster.deprojection_bias_flat`
    - :func:`pymaster.compute_coupled_cell_flat`
    - :func:`pymaster.NmtWorkspaceFlat.decouple_cell`

    :param NmtFieldFlat f1,f2: fields to correlate
    :param NmtBinFlat b: binning scheme defining output bandpower
    :param cl_noise: noise bias (i.e. angular power spectrum of masked noise realizations) (optional).  This power spectrum should be defined at the multipoles returned by the method :func:`get_ell_sampling` of either f1 or f2.
    :param cl_guess: set of power spectra corresponding to a best-guess of the true power spectra of f1 and f2. Needed only to compute the contaminant cleaning bias (optional).
    :param ells_guess: multipoles at which cl_guess is defined.
    :param NmtWorkspaceFlat workspace: object containing the mode-coupling matrix associated with an incomplete sky coverage. If provided, the function will skip the computation of the mode-coupling matrix and use the information encoded in this object.
    :param int nell_rebin: number of sub-intervals into which the base k-intervals will be sub-sampled to compute the coupling matrix
    :param float(2) ell_cut_x: remove all modes with ell_x in the interval [ell_cut_x[0],ell_cut_x[1]] from the calculation.
    :param float(2) ell_cut_y: remove all modes with ell_y in the interval [ell_cut_y[0],ell_cut_y[1]] from the calculation.
    :return: set of decoupled bandpowers
    """
    if((f1.nx!=f2.nx) or (f1.ny!=f2.ny)) :
        raise KeyError("Fields must have same resolution")
    if cl_noise!=None :
        if(len(cl_noise)!=f1.fl.nmaps*f2.fl.nmaps) :
            raise KeyError("Wrong length for noise power spectrum")
        cln=cl_noise.copy()
    else :
        cln=np.zeros([f1.fl.nmaps*f2.fl.nmaps,f1.fl.fs.n_ell])
    if cl_guess!=None :
        if((len(cl_guess)!=f1.fl.nmaps*f2.fl.nmaps) or (len(cl_guess[0])!=len(ells_guess))) :
            raise KeyError("Wrong length for guess power spectrum")
        lf=ells_guess.copy()
        clg=cl_guess.copy()
    else :
        lf=(np.arange(f1.fl.fs.n_ell)+0.5)*f1.fl.fs.dell
        clg=np.zeros([f1.fl.nmaps*f2.fl.nmaps,f1.fl.fs.n_ell])

    if workspace==None :
        cl1d=lib.comp_pspec_flat(f1.fl,f2.fl,b.bin,None,nell_rebin,method,
                                 cln,lf,clg,len(cln)*b.bin.n_bands,
                                 ell_cut_x[0],ell_cut_x[1],ell_cut_y[0],ell_cut_y[1])
    else :
        cl1d=lib.comp_pspec_flat(f1.fl,f2.fl,b.bin,workspace.wsp,nell_rebin,method,
                                 cln,lf,clg,len(cln)*b.bin.n_bands,
                                 ell_cut_x[0],ell_cut_x[1],ell_cut_y[0],ell_cut_y[1])

    clout=np.reshape(cl1d,[len(cln),b.bin.n_bands])

    return clout
