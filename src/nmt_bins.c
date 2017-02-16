#include "utils.h"

void nmt_bins_free(nmt_binning_scheme *bins)
{
  int ii;
  free(bins->nell_list);
  for(ii=0;ii<bins->n_bands;ii++) {
    free(bins->ell_list[ii]);
    free(bins->w_list[ii]);
  }
  free(bins->ell_list);
  free(bins->w_list);
}

nmt_binning_scheme *nmt_bins_create(int nlb,int lmax)
{
  int ii;
  int nband_max=(lmax-1)/nlb;
  flouble w0=1./nlb;

  nmt_binning_scheme *bins=my_malloc(sizeof(nmt_binning_scheme));
  bins->n_bands=nband_max;
  bins->nell_list=my_calloc(nband_max,sizeof(int));
  bins->ell_list=my_malloc(nband_max*sizeof(int *));
  bins->w_list=my_malloc(nband_max*sizeof(flouble *));

  for(ii=0;ii<nband_max;ii++) {
    int jj;
    bins->nell_list[ii]=nlb;
    bins->ell_list[ii]=my_malloc(nlb*sizeof(int));
    bins->w_list[ii]=my_malloc(nlb*sizeof(flouble));
    for(jj=0;jj<nlb;jj++) {
      bins->ell_list[ii][jj]=2+ii*nlb+jj;
      bins->w_list[ii][jj]=w0;
    }
  }

  return bins;
}

nmt_binning_scheme *nmt_bins_read(char *fname,int lmax)
{
  FILE *fi=my_fopen(fname,"r");
  int ii,nlines=my_linecount(fi); rewind(fi);
  if(nlines!=lmax+1)
    report_error(1,"Error reading binning table\n");

  int *band_number,*larr;
  flouble *warr;
  band_number=my_malloc(nlines*sizeof(int));
  larr=my_malloc(nlines*sizeof(int));
  warr=my_malloc(nlines*sizeof(flouble));
  
  int nband_max=0;
  for(ii=0;ii<nlines;ii++) {
    double w;
    int stat=fscanf(fi,"%d %d %lf",&(band_number[ii]),&(larr[ii]),&w);
    if(stat!=3)
      report_error(1,"Error reading %s, line %d\n",fname,ii+1);
    warr[ii]=w;
    if(band_number[ii]>nband_max)
      nband_max=band_number[ii];
  }
  fclose(fi);
  nband_max++;

  nmt_binning_scheme *bins=my_malloc(sizeof(nmt_binning_scheme));
  bins->n_bands=nband_max;
  bins->nell_list=my_calloc(nband_max,sizeof(int));
  bins->ell_list=my_malloc(nband_max*sizeof(int *));
  bins->w_list=my_malloc(nband_max*sizeof(flouble *));
  
  for(ii=0;ii<nlines;ii++) {
    if(band_number[ii]>=0)
      bins->nell_list[band_number[ii]]++;
  }

  for(ii=0;ii<nband_max;ii++) {
    bins->ell_list[ii]=my_malloc(bins->nell_list[ii]*sizeof(int));
    bins->w_list[ii]=my_malloc(bins->nell_list[ii]*sizeof(flouble));
  }

  for(ii=0;ii<nlines;ii++)
    bins->nell_list[band_number[ii]]=0;

  for(ii=0;ii<nlines;ii++) {
    int l=larr[ii];
    int b=band_number[ii];
    flouble w=warr[ii];

    if(b>=0) {
      bins->ell_list[b][bins->nell_list[b]]=l;
      bins->w_list[b][bins->nell_list[b]]=w;
      bins->nell_list[b]++;
    }
  }

  for(ii=0;ii<nband_max;ii++) {
    int jj;
    flouble norm=0;
    for(jj=0;jj<bins->nell_list[ii];jj++)
      norm+=bins->w_list[ii][jj];
    if(norm<=0)
      report_error(1,"Weights in band %d are wrong\n",ii);
    for(jj=0;jj<bins->nell_list[ii];jj++)
      bins->w_list[ii][jj]/=norm;
  }

  free(larr);
  free(band_number);
  free(warr);

  return bins;
}

void nmt_bin_cls(nmt_binning_scheme *bin,flouble **cls_in,flouble **cls_out,int ncls)
{
  int icl;

  for(icl=0;icl<ncls;icl++) {
    int ib;
    for(ib=0;ib<bin->n_bands;ib++) {
      int il;
      cls_out[icl][ib]=0;
      for(il=0;il<bin->nell_list[ib];il++) {
	int l=bin->ell_list[ib][il];
	flouble w=bin->w_list[ib][il];
	cls_out[icl][ib]+=w*cls_in[icl][l];
      }
    }
  }
}

void nmt_unbin_cls(nmt_binning_scheme *bin,flouble **cls_in,flouble **cls_out,int ncls)
{
  int icl;

  for(icl=0;icl<ncls;icl++) {
    int ib;
    for(ib=0;ib<bin->n_bands;ib++) {
      int il;
      flouble clb=cls_in[icl][ib];
      for(il=0;il<bin->nell_list[ib];il++) {
	int l=bin->ell_list[ib][il];
	cls_out[icl][l]=clb;
      }
    }
  }
}

void nmt_ell_eff(nmt_binning_scheme *bin,flouble *larr)
{
  int ib;

  for(ib=0;ib<bin->n_bands;ib++) {
    int il;
    larr[ib]=0;
    for(il=0;il<bin->nell_list[ib];il++)
      larr[ib]+=bin->ell_list[ib][il]*bin->w_list[ib][il];
  }
}
