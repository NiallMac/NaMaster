#include "utils.h"

int nmt_bins_flat_search_fast(nmt_binning_scheme_flat *bin,flouble l,int il)
{
  if(l<bin->ell_0_list[il]) {
    int ilback=il-1;
    while(ilback>=0) {
      if(l<bin->ell_0_list[ilback])
	ilback--;
      else
	return ilback;
    }
    return -1;
  }
  else if(l>=bin->ell_f_list[il]) {
    int ilback=il+1;
    while(ilback<bin->n_bands) {
      if(l>bin->ell_f_list[ilback])
	ilback++;
      else
	return ilback;
    }
    return -1;
  }
  else {
    return il;
  }
}

void nmt_bins_flat_free(nmt_binning_scheme_flat *bin)
{
  free(bin->ell_0_list);
  free(bin->ell_f_list);
  free(bin);
}

nmt_binning_scheme_flat *nmt_bins_flat_constant(int nlb,flouble lmax)
{
  int ii;
  int nband_max=(int)((lmax-1)/nlb);

  nmt_binning_scheme_flat *bin=my_malloc(sizeof(nmt_binning_scheme_flat));
  bin->n_bands=nband_max;
  bin->ell_0_list=my_calloc(nband_max,sizeof(flouble));
  bin->ell_f_list=my_calloc(nband_max,sizeof(flouble));

  for(ii=0;ii<nband_max;ii++) {
    bin->ell_0_list[ii]=2+nlb*ii;
    bin->ell_f_list[ii]=2+nlb*(ii+1)-1;
  }

  return bin;
}

nmt_binning_scheme_flat *nmt_bins_flat_create(int nell,flouble *l0,flouble *lf)
{
  nmt_binning_scheme_flat *bin=my_malloc(sizeof(nmt_binning_scheme_flat));
  bin->n_bands=nell;
  bin->ell_0_list=my_calloc(nell,sizeof(flouble));
  bin->ell_f_list=my_calloc(nell,sizeof(flouble));
  memcpy(bin->ell_0_list,l0,nell*sizeof(flouble));
  memcpy(bin->ell_f_list,lf,nell*sizeof(flouble));

  return bin;
}

void nmt_bin_cls_flat(nmt_binning_scheme_flat *bin,int nl,flouble *larr,flouble **cls_in,
		      flouble **cls_out,int ncls)
{
  int icl;
  int *nells=my_malloc(bin->n_bands*sizeof(int));

  for(icl=0;icl<ncls;icl++) {
    int ib,il;
    for(ib=0;ib<bin->n_bands;ib++) {
      cls_out[icl][ib]=0;
      nells[ib]=0;
    }

    ib=0;
    for(il=0;il<nl;il++) {
      ib=nmt_bins_flat_search_fast(bin,larr[il],ib);
      cls_out[icl][ib]+=cls_in[icl][il];
      nells[ib]++;
    }

    for(ib=0;ib<bin->n_bands;ib++) {
      if(nells[ib]>0)
	cls_out[icl][ib]/=nells[ib];
    }
  }
}

void nmt_unbin_cls_flat(nmt_binning_scheme_flat *bin,flouble **cls_in,
			int nl,flouble *larr,flouble **cls_out,int ncls)
{
  int icl;

  for(icl=0;icl<ncls;icl++) {
    int il,ib=0;
    for(il=0;il<nl;il++) {
      ib=nmt_bins_flat_search_fast(bin,larr[il],ib);
      cls_out[icl][il]=cls_in[icl][ib];
    }
  }
}

void nmt_ell_eff_flat(nmt_binning_scheme_flat *bin,flouble *larr)
{
  int ib;

  for(ib=0;ib<bin->n_bands;ib++)
    larr[ib]=0.5*(bin->ell_0_list[ib]+bin->ell_f_list[ib]);
}
