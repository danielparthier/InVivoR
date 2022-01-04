
// function to apply FIR filter to data using convolution and zero-padding
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
arma::vec FirFilteringInternal(const arma::colvec& SIGNAL,
                                  const arma::colvec& FIR_FILTER,
                                  bool FiltFilt = true,
                                  unsigned int BatchSize = 1e4,
                                  bool padding = true,
                                  const int& CORES = 1) {
  omp_set_num_threads(CORES);
  unsigned int ChunkLength;
  if(BatchSize< 6*FIR_FILTER.n_elem) {
    BatchSize = 6*FIR_FILTER.n_elem;
    ChunkLength = std::pow(2, std::ceil(std::log2(BatchSize)));
    BatchSize = ChunkLength-FIR_FILTER.n_elem*4;
  } else {
    ChunkLength = std::pow(2, std::ceil(std::log2(BatchSize)));
    BatchSize = ChunkLength-FIR_FILTER.n_elem*4;
  }
  if(ChunkLength>SIGNAL.size()) {
    ChunkLength = SIGNAL.n_elem+FIR_FILTER.n_elem*6;
    BatchSize = SIGNAL.n_elem;
  } else if(ChunkLength < 6*FIR_FILTER.size()) {
    ChunkLength = std::pow(2, std::ceil(std::log2(6*FIR_FILTER.size())));
    BatchSize = ChunkLength-FIR_FILTER.n_elem*4;
  }
  arma::vec OutPutVec;
  if(padding) {
    OutPutVec = arma::zeros<arma::vec>(SIGNAL.size()+10*FIR_FILTER.size());
  } else {
    OutPutVec = arma::zeros<arma::vec>(SIGNAL.size());
  }
  arma::vec FilterPadded = arma::zeros<arma::vec>(ChunkLength);
  FilterPadded.subvec(0, size(FIR_FILTER)) = FIR_FILTER;
  arma::cx_mat Filter_FFT = arma::fft(FilterPadded);
  unsigned int BatchJump = BatchSize;
  //index for padding end at beginning
  unsigned int prePaddingArea=(ChunkLength-BatchSize)/2;//prePaddingArea = 5*FIR_FILTER.size()-1;
#pragma omp parallel for shared(padding, prePaddingArea, OutPutVec, SIGNAL,BatchSize, ChunkLength, Filter_FFT, BatchJump) schedule(dynamic) default(none)
  for(unsigned long int i = 0; i<OutPutVec.size(); i += BatchJump) {
    arma::vec InputSignalPadded = arma::zeros<arma::vec>(ChunkLength);
    unsigned long int subBegin;
    unsigned long int subEnd;
    unsigned long int padBegin;
    if((i>prePaddingArea) & (i<(SIGNAL.n_elem-BatchSize-1))) {
      // outside padding area (begin and end)
      subBegin = i-prePaddingArea;
      subEnd = i+BatchSize-prePaddingArea;
      padBegin = prePaddingArea-1;
    } else if(((i+BatchSize)>prePaddingArea) & (i<(SIGNAL.n_elem-BatchSize-1))) {
      // still in padding beginning
      subBegin = 0;
      subEnd = i+BatchSize-prePaddingArea;
      padBegin = ChunkLength-prePaddingArea-subEnd;
    } else if(((i+BatchSize+prePaddingArea)>SIGNAL.n_elem) & ((i-prePaddingArea)<(SIGNAL.n_elem-1))) {
      // end of chunk in padding area
      subBegin = i-prePaddingArea;
      subEnd = SIGNAL.n_elem-1;
      padBegin = prePaddingArea-1;
    } else {
      continue;
    }
    InputSignalPadded.subvec(padBegin,size(SIGNAL.subvec(subBegin, subEnd))) = SIGNAL.subvec(subBegin, subEnd);
    InputSignalPadded = arma::real(arma::ifft(arma::fft(InputSignalPadded)%Filter_FFT));
#pragma omp critical
    if((InputSignalPadded.n_elem+i)>OutPutVec.n_elem) {
      // if overshoot trace
      OutPutVec.subvec(i, OutPutVec.n_elem-1) += InputSignalPadded.subvec(0,(OutPutVec.n_elem-1-i));
    } else {
      OutPutVec.subvec(i, size(InputSignalPadded)) += InputSignalPadded;  
    }
  }
  if(FiltFilt) {
    OutPutVec = FirFilteringInternal(arma::reverse(OutPutVec), FIR_FILTER, false, BatchSize, false, CORES);
    OutPutVec = arma::reverse(OutPutVec);
    return OutPutVec.subvec(0,size(SIGNAL));
  }
  if(!padding) {
    return OutPutVec; 
  } else {
    return OutPutVec.subvec(prePaddingArea, size(SIGNAL));
  }
}

//' @title FIR filtering
//' 
//' @description This function applies an FIR filter to a signal an returns the filtered trace.
//'
//' @name FirFiltering
//' @param SIGNAL A numeric vector.
//' @param FIR_FILTER A numeric vector which can be used as FIR filter.
//' @param FiltFilt A bool indicating if "filtfilt" mode should be used.
//' @param BatchSize An integer indicating the starting batchsize of the trace (chunk size will be optimised for FFT).
//' @param CORES An integer indicating what number of cores should be used.
//' 
//' @return Returns numeric vector which is the FIR filtered original signal.
//' @export
// [[Rcpp::export]]
arma::vec FirFiltering(const arma::colvec& SIGNAL,
                          const arma::colvec& FIR_FILTER,
                          bool FiltFilt = true,
                          unsigned int BatchSize = 1e4,
                          const int& CORES = 1) {
  if(SIGNAL.size()<BatchSize) {
    BatchSize = SIGNAL.size();
  }
  
  //  return Rcpp::NumericVector(OutPutVec.subvec(prePaddingArea, size(SIGNAL)).begin(),OutPutVec.subvec(prePaddingArea,size(SIGNAL)).end());
  return FirFilteringInternal(SIGNAL, FIR_FILTER, FiltFilt, BatchSize, true, CORES);
}