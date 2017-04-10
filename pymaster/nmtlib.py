# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.11
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_nmtlib', [dirname(__file__)])
        except ImportError:
            import _nmtlib
            return _nmtlib
        if fp is not None:
            try:
                _mod = imp.load_module('_nmtlib', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _nmtlib = swig_import_helper()
    del swig_import_helper
else:
    import _nmtlib
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class binning_scheme(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, binning_scheme, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, binning_scheme, name)
    __repr__ = _swig_repr
    __swig_setmethods__["n_bands"] = _nmtlib.binning_scheme_n_bands_set
    __swig_getmethods__["n_bands"] = _nmtlib.binning_scheme_n_bands_get
    if _newclass:n_bands = _swig_property(_nmtlib.binning_scheme_n_bands_get, _nmtlib.binning_scheme_n_bands_set)
    __swig_setmethods__["nell_list"] = _nmtlib.binning_scheme_nell_list_set
    __swig_getmethods__["nell_list"] = _nmtlib.binning_scheme_nell_list_get
    if _newclass:nell_list = _swig_property(_nmtlib.binning_scheme_nell_list_get, _nmtlib.binning_scheme_nell_list_set)
    __swig_setmethods__["ell_list"] = _nmtlib.binning_scheme_ell_list_set
    __swig_getmethods__["ell_list"] = _nmtlib.binning_scheme_ell_list_get
    if _newclass:ell_list = _swig_property(_nmtlib.binning_scheme_ell_list_get, _nmtlib.binning_scheme_ell_list_set)
    __swig_setmethods__["w_list"] = _nmtlib.binning_scheme_w_list_set
    __swig_getmethods__["w_list"] = _nmtlib.binning_scheme_w_list_get
    if _newclass:w_list = _swig_property(_nmtlib.binning_scheme_w_list_get, _nmtlib.binning_scheme_w_list_set)
    def __init__(self): 
        this = _nmtlib.new_binning_scheme()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _nmtlib.delete_binning_scheme
    __del__ = lambda self : None;
binning_scheme_swigregister = _nmtlib.binning_scheme_swigregister
binning_scheme_swigregister(binning_scheme)


def bins_constant(*args):
  return _nmtlib.bins_constant(*args)
bins_constant = _nmtlib.bins_constant

def bins_create(*args):
  return _nmtlib.bins_create(*args)
bins_create = _nmtlib.bins_create

def bins_read(*args):
  return _nmtlib.bins_read(*args)
bins_read = _nmtlib.bins_read

def bins_free(*args):
  return _nmtlib.bins_free(*args)
bins_free = _nmtlib.bins_free

def bin_cls(*args):
  return _nmtlib.bin_cls(*args)
bin_cls = _nmtlib.bin_cls

def unbin_cls(*args):
  return _nmtlib.unbin_cls(*args)
unbin_cls = _nmtlib.unbin_cls

def ell_eff(*args):
  return _nmtlib.ell_eff(*args)
ell_eff = _nmtlib.ell_eff
class k_function(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, k_function, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, k_function, name)
    __repr__ = _swig_repr
    __swig_setmethods__["is_const"] = _nmtlib.k_function_is_const_set
    __swig_getmethods__["is_const"] = _nmtlib.k_function_is_const_get
    if _newclass:is_const = _swig_property(_nmtlib.k_function_is_const_get, _nmtlib.k_function_is_const_set)
    __swig_setmethods__["nk"] = _nmtlib.k_function_nk_set
    __swig_getmethods__["nk"] = _nmtlib.k_function_nk_get
    if _newclass:nk = _swig_property(_nmtlib.k_function_nk_get, _nmtlib.k_function_nk_set)
    __swig_setmethods__["x0"] = _nmtlib.k_function_x0_set
    __swig_getmethods__["x0"] = _nmtlib.k_function_x0_get
    if _newclass:x0 = _swig_property(_nmtlib.k_function_x0_get, _nmtlib.k_function_x0_set)
    __swig_setmethods__["xf"] = _nmtlib.k_function_xf_set
    __swig_getmethods__["xf"] = _nmtlib.k_function_xf_get
    if _newclass:xf = _swig_property(_nmtlib.k_function_xf_get, _nmtlib.k_function_xf_set)
    __swig_setmethods__["y0"] = _nmtlib.k_function_y0_set
    __swig_getmethods__["y0"] = _nmtlib.k_function_y0_get
    if _newclass:y0 = _swig_property(_nmtlib.k_function_y0_get, _nmtlib.k_function_y0_set)
    __swig_setmethods__["yf"] = _nmtlib.k_function_yf_set
    __swig_getmethods__["yf"] = _nmtlib.k_function_yf_get
    if _newclass:yf = _swig_property(_nmtlib.k_function_yf_get, _nmtlib.k_function_yf_set)
    __swig_setmethods__["spl"] = _nmtlib.k_function_spl_set
    __swig_getmethods__["spl"] = _nmtlib.k_function_spl_get
    if _newclass:spl = _swig_property(_nmtlib.k_function_spl_get, _nmtlib.k_function_spl_set)
    def __init__(self): 
        this = _nmtlib.new_k_function()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _nmtlib.delete_k_function
    __del__ = lambda self : None;
k_function_swigregister = _nmtlib.k_function_swigregister
k_function_swigregister(k_function)


def k_function_alloc(*args):
  return _nmtlib.k_function_alloc(*args)
k_function_alloc = _nmtlib.k_function_alloc

def k_function_free(*args):
  return _nmtlib.k_function_free(*args)
k_function_free = _nmtlib.k_function_free

def k_function_eval(*args):
  return _nmtlib.k_function_eval(*args)
k_function_eval = _nmtlib.k_function_eval
class flatsky_info(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, flatsky_info, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, flatsky_info, name)
    __repr__ = _swig_repr
    __swig_setmethods__["nx"] = _nmtlib.flatsky_info_nx_set
    __swig_getmethods__["nx"] = _nmtlib.flatsky_info_nx_get
    if _newclass:nx = _swig_property(_nmtlib.flatsky_info_nx_get, _nmtlib.flatsky_info_nx_set)
    __swig_setmethods__["ny"] = _nmtlib.flatsky_info_ny_set
    __swig_getmethods__["ny"] = _nmtlib.flatsky_info_ny_get
    if _newclass:ny = _swig_property(_nmtlib.flatsky_info_ny_get, _nmtlib.flatsky_info_ny_set)
    __swig_setmethods__["npix"] = _nmtlib.flatsky_info_npix_set
    __swig_getmethods__["npix"] = _nmtlib.flatsky_info_npix_get
    if _newclass:npix = _swig_property(_nmtlib.flatsky_info_npix_get, _nmtlib.flatsky_info_npix_set)
    __swig_setmethods__["lx"] = _nmtlib.flatsky_info_lx_set
    __swig_getmethods__["lx"] = _nmtlib.flatsky_info_lx_get
    if _newclass:lx = _swig_property(_nmtlib.flatsky_info_lx_get, _nmtlib.flatsky_info_lx_set)
    __swig_setmethods__["ly"] = _nmtlib.flatsky_info_ly_set
    __swig_getmethods__["ly"] = _nmtlib.flatsky_info_ly_get
    if _newclass:ly = _swig_property(_nmtlib.flatsky_info_ly_get, _nmtlib.flatsky_info_ly_set)
    __swig_setmethods__["pixsize"] = _nmtlib.flatsky_info_pixsize_set
    __swig_getmethods__["pixsize"] = _nmtlib.flatsky_info_pixsize_get
    if _newclass:pixsize = _swig_property(_nmtlib.flatsky_info_pixsize_get, _nmtlib.flatsky_info_pixsize_set)
    __swig_setmethods__["n_ell"] = _nmtlib.flatsky_info_n_ell_set
    __swig_getmethods__["n_ell"] = _nmtlib.flatsky_info_n_ell_get
    if _newclass:n_ell = _swig_property(_nmtlib.flatsky_info_n_ell_get, _nmtlib.flatsky_info_n_ell_set)
    __swig_setmethods__["i_dell"] = _nmtlib.flatsky_info_i_dell_set
    __swig_getmethods__["i_dell"] = _nmtlib.flatsky_info_i_dell_get
    if _newclass:i_dell = _swig_property(_nmtlib.flatsky_info_i_dell_get, _nmtlib.flatsky_info_i_dell_set)
    __swig_setmethods__["dell"] = _nmtlib.flatsky_info_dell_set
    __swig_getmethods__["dell"] = _nmtlib.flatsky_info_dell_get
    if _newclass:dell = _swig_property(_nmtlib.flatsky_info_dell_get, _nmtlib.flatsky_info_dell_set)
    __swig_setmethods__["ell_min"] = _nmtlib.flatsky_info_ell_min_set
    __swig_getmethods__["ell_min"] = _nmtlib.flatsky_info_ell_min_get
    if _newclass:ell_min = _swig_property(_nmtlib.flatsky_info_ell_min_get, _nmtlib.flatsky_info_ell_min_set)
    __swig_setmethods__["n_cells"] = _nmtlib.flatsky_info_n_cells_set
    __swig_getmethods__["n_cells"] = _nmtlib.flatsky_info_n_cells_get
    if _newclass:n_cells = _swig_property(_nmtlib.flatsky_info_n_cells_get, _nmtlib.flatsky_info_n_cells_set)
    def __init__(self): 
        this = _nmtlib.new_flatsky_info()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _nmtlib.delete_flatsky_info
    __del__ = lambda self : None;
flatsky_info_swigregister = _nmtlib.flatsky_info_swigregister
flatsky_info_swigregister(flatsky_info)


def flatsky_info_alloc(*args):
  return _nmtlib.flatsky_info_alloc(*args)
flatsky_info_alloc = _nmtlib.flatsky_info_alloc

def flatsky_info_free(*args):
  return _nmtlib.flatsky_info_free(*args)
flatsky_info_free = _nmtlib.flatsky_info_free
class field_flat(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, field_flat, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, field_flat, name)
    __repr__ = _swig_repr
    __swig_setmethods__["fs"] = _nmtlib.field_flat_fs_set
    __swig_getmethods__["fs"] = _nmtlib.field_flat_fs_get
    if _newclass:fs = _swig_property(_nmtlib.field_flat_fs_get, _nmtlib.field_flat_fs_set)
    __swig_setmethods__["npix"] = _nmtlib.field_flat_npix_set
    __swig_getmethods__["npix"] = _nmtlib.field_flat_npix_get
    if _newclass:npix = _swig_property(_nmtlib.field_flat_npix_get, _nmtlib.field_flat_npix_set)
    __swig_setmethods__["pure_e"] = _nmtlib.field_flat_pure_e_set
    __swig_getmethods__["pure_e"] = _nmtlib.field_flat_pure_e_get
    if _newclass:pure_e = _swig_property(_nmtlib.field_flat_pure_e_get, _nmtlib.field_flat_pure_e_set)
    __swig_setmethods__["pure_b"] = _nmtlib.field_flat_pure_b_set
    __swig_getmethods__["pure_b"] = _nmtlib.field_flat_pure_b_get
    if _newclass:pure_b = _swig_property(_nmtlib.field_flat_pure_b_get, _nmtlib.field_flat_pure_b_set)
    __swig_setmethods__["mask"] = _nmtlib.field_flat_mask_set
    __swig_getmethods__["mask"] = _nmtlib.field_flat_mask_get
    if _newclass:mask = _swig_property(_nmtlib.field_flat_mask_get, _nmtlib.field_flat_mask_set)
    __swig_setmethods__["pol"] = _nmtlib.field_flat_pol_set
    __swig_getmethods__["pol"] = _nmtlib.field_flat_pol_get
    if _newclass:pol = _swig_property(_nmtlib.field_flat_pol_get, _nmtlib.field_flat_pol_set)
    __swig_setmethods__["nmaps"] = _nmtlib.field_flat_nmaps_set
    __swig_getmethods__["nmaps"] = _nmtlib.field_flat_nmaps_get
    if _newclass:nmaps = _swig_property(_nmtlib.field_flat_nmaps_get, _nmtlib.field_flat_nmaps_set)
    __swig_setmethods__["maps"] = _nmtlib.field_flat_maps_set
    __swig_getmethods__["maps"] = _nmtlib.field_flat_maps_get
    if _newclass:maps = _swig_property(_nmtlib.field_flat_maps_get, _nmtlib.field_flat_maps_set)
    __swig_setmethods__["alms"] = _nmtlib.field_flat_alms_set
    __swig_getmethods__["alms"] = _nmtlib.field_flat_alms_get
    if _newclass:alms = _swig_property(_nmtlib.field_flat_alms_get, _nmtlib.field_flat_alms_set)
    __swig_setmethods__["ntemp"] = _nmtlib.field_flat_ntemp_set
    __swig_getmethods__["ntemp"] = _nmtlib.field_flat_ntemp_get
    if _newclass:ntemp = _swig_property(_nmtlib.field_flat_ntemp_get, _nmtlib.field_flat_ntemp_set)
    __swig_setmethods__["temp"] = _nmtlib.field_flat_temp_set
    __swig_getmethods__["temp"] = _nmtlib.field_flat_temp_get
    if _newclass:temp = _swig_property(_nmtlib.field_flat_temp_get, _nmtlib.field_flat_temp_set)
    __swig_setmethods__["a_temp"] = _nmtlib.field_flat_a_temp_set
    __swig_getmethods__["a_temp"] = _nmtlib.field_flat_a_temp_get
    if _newclass:a_temp = _swig_property(_nmtlib.field_flat_a_temp_get, _nmtlib.field_flat_a_temp_set)
    __swig_setmethods__["matrix_M"] = _nmtlib.field_flat_matrix_M_set
    __swig_getmethods__["matrix_M"] = _nmtlib.field_flat_matrix_M_get
    if _newclass:matrix_M = _swig_property(_nmtlib.field_flat_matrix_M_get, _nmtlib.field_flat_matrix_M_set)
    __swig_setmethods__["beam"] = _nmtlib.field_flat_beam_set
    __swig_getmethods__["beam"] = _nmtlib.field_flat_beam_get
    if _newclass:beam = _swig_property(_nmtlib.field_flat_beam_get, _nmtlib.field_flat_beam_set)
    def __init__(self): 
        this = _nmtlib.new_field_flat()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _nmtlib.delete_field_flat
    __del__ = lambda self : None;
field_flat_swigregister = _nmtlib.field_flat_swigregister
field_flat_swigregister(field_flat)


def field_flat_free(*args):
  return _nmtlib.field_flat_free(*args)
field_flat_free = _nmtlib.field_flat_free

def field_flat_alloc(*args):
  return _nmtlib.field_flat_alloc(*args)
field_flat_alloc = _nmtlib.field_flat_alloc

def synfast_flat(*args):
  return _nmtlib.synfast_flat(*args)
synfast_flat = _nmtlib.synfast_flat
class field(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, field, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, field, name)
    __repr__ = _swig_repr
    __swig_setmethods__["nside"] = _nmtlib.field_nside_set
    __swig_getmethods__["nside"] = _nmtlib.field_nside_get
    if _newclass:nside = _swig_property(_nmtlib.field_nside_get, _nmtlib.field_nside_set)
    __swig_setmethods__["npix"] = _nmtlib.field_npix_set
    __swig_getmethods__["npix"] = _nmtlib.field_npix_get
    if _newclass:npix = _swig_property(_nmtlib.field_npix_get, _nmtlib.field_npix_set)
    __swig_setmethods__["lmax"] = _nmtlib.field_lmax_set
    __swig_getmethods__["lmax"] = _nmtlib.field_lmax_get
    if _newclass:lmax = _swig_property(_nmtlib.field_lmax_get, _nmtlib.field_lmax_set)
    __swig_setmethods__["pure_e"] = _nmtlib.field_pure_e_set
    __swig_getmethods__["pure_e"] = _nmtlib.field_pure_e_get
    if _newclass:pure_e = _swig_property(_nmtlib.field_pure_e_get, _nmtlib.field_pure_e_set)
    __swig_setmethods__["pure_b"] = _nmtlib.field_pure_b_set
    __swig_getmethods__["pure_b"] = _nmtlib.field_pure_b_get
    if _newclass:pure_b = _swig_property(_nmtlib.field_pure_b_get, _nmtlib.field_pure_b_set)
    __swig_setmethods__["mask"] = _nmtlib.field_mask_set
    __swig_getmethods__["mask"] = _nmtlib.field_mask_get
    if _newclass:mask = _swig_property(_nmtlib.field_mask_get, _nmtlib.field_mask_set)
    __swig_setmethods__["pol"] = _nmtlib.field_pol_set
    __swig_getmethods__["pol"] = _nmtlib.field_pol_get
    if _newclass:pol = _swig_property(_nmtlib.field_pol_get, _nmtlib.field_pol_set)
    __swig_setmethods__["nmaps"] = _nmtlib.field_nmaps_set
    __swig_getmethods__["nmaps"] = _nmtlib.field_nmaps_get
    if _newclass:nmaps = _swig_property(_nmtlib.field_nmaps_get, _nmtlib.field_nmaps_set)
    __swig_setmethods__["maps"] = _nmtlib.field_maps_set
    __swig_getmethods__["maps"] = _nmtlib.field_maps_get
    if _newclass:maps = _swig_property(_nmtlib.field_maps_get, _nmtlib.field_maps_set)
    __swig_setmethods__["alms"] = _nmtlib.field_alms_set
    __swig_getmethods__["alms"] = _nmtlib.field_alms_get
    if _newclass:alms = _swig_property(_nmtlib.field_alms_get, _nmtlib.field_alms_set)
    __swig_setmethods__["ntemp"] = _nmtlib.field_ntemp_set
    __swig_getmethods__["ntemp"] = _nmtlib.field_ntemp_get
    if _newclass:ntemp = _swig_property(_nmtlib.field_ntemp_get, _nmtlib.field_ntemp_set)
    __swig_setmethods__["temp"] = _nmtlib.field_temp_set
    __swig_getmethods__["temp"] = _nmtlib.field_temp_get
    if _newclass:temp = _swig_property(_nmtlib.field_temp_get, _nmtlib.field_temp_set)
    __swig_setmethods__["a_temp"] = _nmtlib.field_a_temp_set
    __swig_getmethods__["a_temp"] = _nmtlib.field_a_temp_get
    if _newclass:a_temp = _swig_property(_nmtlib.field_a_temp_get, _nmtlib.field_a_temp_set)
    __swig_setmethods__["matrix_M"] = _nmtlib.field_matrix_M_set
    __swig_getmethods__["matrix_M"] = _nmtlib.field_matrix_M_get
    if _newclass:matrix_M = _swig_property(_nmtlib.field_matrix_M_get, _nmtlib.field_matrix_M_set)
    __swig_setmethods__["beam"] = _nmtlib.field_beam_set
    __swig_getmethods__["beam"] = _nmtlib.field_beam_get
    if _newclass:beam = _swig_property(_nmtlib.field_beam_get, _nmtlib.field_beam_set)
    def __init__(self): 
        this = _nmtlib.new_field()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _nmtlib.delete_field
    __del__ = lambda self : None;
field_swigregister = _nmtlib.field_swigregister
field_swigregister(field)


def field_free(*args):
  return _nmtlib.field_free(*args)
field_free = _nmtlib.field_free

def field_alloc_sph(*args):
  return _nmtlib.field_alloc_sph(*args)
field_alloc_sph = _nmtlib.field_alloc_sph

def field_read(*args):
  return _nmtlib.field_read(*args)
field_read = _nmtlib.field_read

def synfast_sph(*args):
  return _nmtlib.synfast_sph(*args)
synfast_sph = _nmtlib.synfast_sph

def apodize_mask(*args):
  return _nmtlib.apodize_mask(*args)
apodize_mask = _nmtlib.apodize_mask
class workspace(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, workspace, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, workspace, name)
    __repr__ = _swig_repr
    __swig_setmethods__["lmax"] = _nmtlib.workspace_lmax_set
    __swig_getmethods__["lmax"] = _nmtlib.workspace_lmax_get
    if _newclass:lmax = _swig_property(_nmtlib.workspace_lmax_get, _nmtlib.workspace_lmax_set)
    __swig_setmethods__["ncls"] = _nmtlib.workspace_ncls_set
    __swig_getmethods__["ncls"] = _nmtlib.workspace_ncls_get
    if _newclass:ncls = _swig_property(_nmtlib.workspace_ncls_get, _nmtlib.workspace_ncls_set)
    __swig_setmethods__["pcl_masks"] = _nmtlib.workspace_pcl_masks_set
    __swig_getmethods__["pcl_masks"] = _nmtlib.workspace_pcl_masks_get
    if _newclass:pcl_masks = _swig_property(_nmtlib.workspace_pcl_masks_get, _nmtlib.workspace_pcl_masks_set)
    __swig_setmethods__["coupling_matrix_unbinned"] = _nmtlib.workspace_coupling_matrix_unbinned_set
    __swig_getmethods__["coupling_matrix_unbinned"] = _nmtlib.workspace_coupling_matrix_unbinned_get
    if _newclass:coupling_matrix_unbinned = _swig_property(_nmtlib.workspace_coupling_matrix_unbinned_get, _nmtlib.workspace_coupling_matrix_unbinned_set)
    __swig_setmethods__["bin"] = _nmtlib.workspace_bin_set
    __swig_getmethods__["bin"] = _nmtlib.workspace_bin_get
    if _newclass:bin = _swig_property(_nmtlib.workspace_bin_get, _nmtlib.workspace_bin_set)
    __swig_setmethods__["coupling_matrix_binned"] = _nmtlib.workspace_coupling_matrix_binned_set
    __swig_getmethods__["coupling_matrix_binned"] = _nmtlib.workspace_coupling_matrix_binned_get
    if _newclass:coupling_matrix_binned = _swig_property(_nmtlib.workspace_coupling_matrix_binned_get, _nmtlib.workspace_coupling_matrix_binned_set)
    __swig_setmethods__["coupling_matrix_perm"] = _nmtlib.workspace_coupling_matrix_perm_set
    __swig_getmethods__["coupling_matrix_perm"] = _nmtlib.workspace_coupling_matrix_perm_get
    if _newclass:coupling_matrix_perm = _swig_property(_nmtlib.workspace_coupling_matrix_perm_get, _nmtlib.workspace_coupling_matrix_perm_set)
    def __init__(self): 
        this = _nmtlib.new_workspace()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _nmtlib.delete_workspace
    __del__ = lambda self : None;
workspace_swigregister = _nmtlib.workspace_swigregister
workspace_swigregister(workspace)


def compute_coupling_matrix(*args):
  return _nmtlib.compute_coupling_matrix(*args)
compute_coupling_matrix = _nmtlib.compute_coupling_matrix

def workspace_write(*args):
  return _nmtlib.workspace_write(*args)
workspace_write = _nmtlib.workspace_write

def workspace_read(*args):
  return _nmtlib.workspace_read(*args)
workspace_read = _nmtlib.workspace_read

def workspace_free(*args):
  return _nmtlib.workspace_free(*args)
workspace_free = _nmtlib.workspace_free

def compute_deprojection_bias(*args):
  return _nmtlib.compute_deprojection_bias(*args)
compute_deprojection_bias = _nmtlib.compute_deprojection_bias

def couple_cl_l(*args):
  return _nmtlib.couple_cl_l(*args)
couple_cl_l = _nmtlib.couple_cl_l

def decouple_cl_l(*args):
  return _nmtlib.decouple_cl_l(*args)
decouple_cl_l = _nmtlib.decouple_cl_l

def compute_coupled_cell(*args):
  return _nmtlib.compute_coupled_cell(*args)
compute_coupled_cell = _nmtlib.compute_coupled_cell

def compute_power_spectra(*args):
  return _nmtlib.compute_power_spectra(*args)
compute_power_spectra = _nmtlib.compute_power_spectra

def get_nell_list(*args):
  return _nmtlib.get_nell_list(*args)
get_nell_list = _nmtlib.get_nell_list

def get_nell(*args):
  return _nmtlib.get_nell(*args)
get_nell = _nmtlib.get_nell

def get_ell_list(*args):
  return _nmtlib.get_ell_list(*args)
get_ell_list = _nmtlib.get_ell_list

def get_weight_list(*args):
  return _nmtlib.get_weight_list(*args)
get_weight_list = _nmtlib.get_weight_list

def get_ell_eff(*args):
  return _nmtlib.get_ell_eff(*args)
get_ell_eff = _nmtlib.get_ell_eff

def bins_create_py(*args):
  return _nmtlib.bins_create_py(*args)
bins_create_py = _nmtlib.bins_create_py

def bin_cl(*args):
  return _nmtlib.bin_cl(*args)
bin_cl = _nmtlib.bin_cl

def unbin_cl(*args):
  return _nmtlib.unbin_cl(*args)
unbin_cl = _nmtlib.unbin_cl

def field_alloc_new(*args):
  return _nmtlib.field_alloc_new(*args)
field_alloc_new = _nmtlib.field_alloc_new

def field_alloc_new_notemp(*args):
  return _nmtlib.field_alloc_new_notemp(*args)
field_alloc_new_notemp = _nmtlib.field_alloc_new_notemp

def field_alloc_new_flat(*args):
  return _nmtlib.field_alloc_new_flat(*args)
field_alloc_new_flat = _nmtlib.field_alloc_new_flat

def field_alloc_new_notemp_flat(*args):
  return _nmtlib.field_alloc_new_notemp_flat(*args)
field_alloc_new_notemp_flat = _nmtlib.field_alloc_new_notemp_flat

def get_map(*args):
  return _nmtlib.get_map(*args)
get_map = _nmtlib.get_map

def get_temp(*args):
  return _nmtlib.get_temp(*args)
get_temp = _nmtlib.get_temp

def apomask(*args):
  return _nmtlib.apomask(*args)
apomask = _nmtlib.apomask

def synfast_new(*args):
  return _nmtlib.synfast_new(*args)
synfast_new = _nmtlib.synfast_new

def synfast_new_flat(*args):
  return _nmtlib.synfast_new_flat(*args)
synfast_new_flat = _nmtlib.synfast_new_flat

def comp_deproj_bias(*args):
  return _nmtlib.comp_deproj_bias(*args)
comp_deproj_bias = _nmtlib.comp_deproj_bias

def comp_pspec_coupled(*args):
  return _nmtlib.comp_pspec_coupled(*args)
comp_pspec_coupled = _nmtlib.comp_pspec_coupled

def decouple_cell_py(*args):
  return _nmtlib.decouple_cell_py(*args)
decouple_cell_py = _nmtlib.decouple_cell_py

def couple_cell_py(*args):
  return _nmtlib.couple_cell_py(*args)
couple_cell_py = _nmtlib.couple_cell_py

def comp_pspec(*args):
  return _nmtlib.comp_pspec(*args)
comp_pspec = _nmtlib.comp_pspec
# This file is compatible with both classic and new-style classes.


