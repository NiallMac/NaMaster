import nmtlib as lib
import numpy as np

class NmtCovarianceWorkspace(object) :
    """
    NmtCovarianceWorkspace objects are used to compute and store the coupling coefficients needed to calculate the Gaussian covariance matrix under the Efstathiou approximation (astro-ph/0307515). When initialized, this object is practically empty. The information describing the coupling coefficients must be computed or read from a file afterwards.
    """
    def __init__(self) :
        self.wsp=None
    
    def __del__(self) :
        if(self.wsp is not None) :
            lib.covar_workspace_free(self.wsp)

    def read_from(self,fname) :
        """
        Reads the contents of an NmtCovarianceWorkspace object from a file (encoded using an internal binary format).

        :param str fname: input file name
        """
        if self.wsp is not None :
            lib.covar_workspace_free(self.wsp)
        self.wsp=lib.covar_workspace_read(fname);

    def compute_coupling_coefficients(self,wa,wb) :
        """
        Computes coupling coefficients of the Gaussian covariance between the power spectra computed using wa and wb (two NmtWorkspace objects).

        :param NmtWorkspace wa,wb: workspaces used to compute the two power spectra whose covariance matrix you want to compute.
        """
        ns=wa.wsp.nside;
        if(wa.wsp.nside!=wb.wsp.nside) :
            raise ValueError("Everything should have the same resolution!")
        if((wa.wsp.ncls!=1) or (wb.wsp.ncls!=1)) :
            raise ValueError("Gaussian covariances only supported for spin-0 fields")
        if self.wsp is not None :
            lib.covar_workspace_free(self.wsp)
        self.wsp=lib.covar_workspace_init(wa.wsp,wb.wsp)

    def write_to(self,fname) :
        """
        Writes the contents of an NmtCovarianceWorkspace object to a file (encoded using an internal binary format).

        :param str fname: output file name
        """
        if self.wsp is None :
            raise KeyError("Must initialize workspace before writing")
        lib.covar_workspace_write(self.wsp,fname)

def gaussian_covariance(cw,cla1b1,cla1b2,cla2b1,cla2b2) :
    """
    Computes Gaussian covariance matrix for power spectra using the information precomputed in cw (a NmtCovarianceWorkspace object). cw should have been initialized using the two NmtWorkspace objects used to compute these power spectra. The notation above assumes that these two NmtWorkspace objects, wa and wb, were used to compute the power spectra of four fields: a1 and a2 for wa, b1 and b2 for wb. Then, claXbY above should be a prediction for the power spectrum between fields aX and bY. These predicted input power spectra should be defined for all ells <=3*nside (where nside is the HEALPix resolution parameter of the fields that were correlated).

    :param NmtCovarianceWorkspace cw: workspaces containing the precomputed coupling coefficients.
    :param cla1b1: prediction for the cross-power spectrum between a1 and b1.
    :param cla1b2: prediction for the cross-power spectrum between a1 and b2.
    :param cla2b1: prediction for the cross-power spectrum between a2 and b1.
    :param cla2b2: prediction for the cross-power spectrum between a2 and b2.
    """
    if((len(cla1b1)!=cw.wsp.lmax_a+1) or (len(cla1b2)!=cw.wsp.lmax_a+1) or (len(cla2b1)!=cw.wsp.lmax_a+1) or (len(cla2b2)!=cw.wsp.lmax_a+1)) :
        raise ValueError("Input C_ls have a weird length")
    len_a=cw.wsp.ncls_a*cw.wsp.bin_a.n_bands
    len_b=cw.wsp.ncls_b*cw.wsp.bin_b.n_bands

    covar1d=lib.comp_gaussian_covariance(cw.wsp,cla1b1,cla1b2,cla2b1,cla2b2,len_a*len_b)
    covar=np.reshape(covar1d,[len_a,len_b])
    return covar
