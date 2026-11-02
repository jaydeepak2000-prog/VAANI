#pragma once
#include <cmath>
#include <cstdint>
#include "frontend_constants.h"

// Exact frontend contract: 400-sample periodic Hann, FFT512 power /512,
// triangular HTK mel bank, natural log, optional orthonormal DCT-II.
class Frontend {
  float real[512],imag[512],power[257],mfe[32];
public:
  void compute(const int16_t* pcm,int8_t* out,float scale,int zero) {
    for(int i=0;i<512;i++){real[i]=i<400?pcm[i]*(1.f/32768.f)*k_window[i]:0;imag[i]=0;}
    for(int i=1,j=0;i<512;i++){
      int bit=256;for(;j&bit;bit>>=1)j^=bit;j^=bit;
      if(i<j){float v=real[i];real[i]=real[j];real[j]=v;}
    }
    for(int len=2;len<=512;len*=2){
      float angle=-6.283185307179586f/len,wr=cosf(angle),wi=sinf(angle);
      for(int start=0;start<512;start+=len){
        float ur=1,ui=0;
        for(int j=0;j<len/2;j++){
          int a=start+j,b=a+len/2;
          float vr=real[b]*ur-imag[b]*ui,vi=real[b]*ui+imag[b]*ur;
          real[b]=real[a]-vr;imag[b]=imag[a]-vi;real[a]+=vr;imag[a]+=vi;
          float next=ur*wr-ui*wi;ui=ur*wi+ui*wr;ur=next;
        }
      }
    }
    for(int i=0;i<257;i++)power[i]=(real[i]*real[i]+imag[i]*imag[i])/512.f;
    for(int b=0;b<32;b++){
      float sum=0;for(int f=0;f<257;f++)sum+=power[f]*k_mel[f][b];
      mfe[b]=logf(fmaxf(sum,1e-10f));
    }
    for(int b=0;b<FEATURE_BINS;b++){
      float value=mfe[b];
      if(USE_MFCC){value=0;for(int j=0;j<32;j++)value+=mfe[j]*k_dct[b][j];}
      float q=roundf(((value-k_mean[b])/k_std[b])/scale)+zero;
      out[b]=(int8_t)fmaxf(-128,fminf(127,q));
    }
  }
};
