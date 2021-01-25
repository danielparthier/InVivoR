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
  if(BatchSize > FIR_FILTER.size()) {
    ChunkLength = std::pow(2, std::ceil(std::log2(BatchSize*1.5)));
    BatchSize = ChunkLength/2;
  } else {
    BatchSize = FIR_FILTER.size();
    ChunkLength = std::pow(2, std::ceil(std::log2(BatchSize*1.5)));
    BatchSize = ChunkLength/2;
  }
  if(ChunkLength>SIGNAL.size()) {
  } else if(ChunkLength < 3*FIR_FILTER.size()) {
    ChunkLength = std::pow(2, std::ceil(std::log2(3*FIR_FILTER.size())));
    BatchSize = ChunkLength/2;
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
  unsigned int prePaddingArea = 5*FIR_FILTER.size()-1;
#pragma omp parallel for shared(padding, prePaddingArea, OutPutVec, SIGNAL,BatchSize, ChunkLength, Filter_FFT, BatchJump) schedule(dynamic) default(none)
  for(unsigned long int i = 0; i<OutPutVec.size(); i += BatchJump) {
    arma::vec InputSignalPadded = arma::zeros<arma::vec>(ChunkLength);
    unsigned long int sig_i = i-prePaddingArea;
    if(i < prePaddingArea) {
      if((i+BatchSize) < prePaddingArea) {
        //do nothing
      } else if((i+BatchSize) > prePaddingArea){
        InputSignalPadded.subvec(prePaddingArea-i, size(SIGNAL.subvec(0, sig_i+BatchSize))) = SIGNAL.subvec(0, sig_i+BatchSize);
      }
    } else if((i>prePaddingArea) & (sig_i < SIGNAL.size())) {
      if(SIGNAL.size() < (sig_i+BatchSize)) {
        BatchSize = SIGNAL.size()-sig_i;
      } 
      InputSignalPadded.subvec(0, size(SIGNAL.subvec(sig_i, sig_i+BatchSize-1))) = SIGNAL.subvec(sig_i, sig_i+BatchSize-1);
    }
    InputSignalPadded = arma::real(arma::ifft(arma::fft(InputSignalPadded)%Filter_FFT));
#pragma omp critical
    if((i+ChunkLength-1) < OutPutVec.size()) {
      OutPutVec.subvec(i, size(InputSignalPadded)) += InputSignalPadded; 
    } else if((i+ChunkLength-1) >= OutPutVec.size()) {
      if(!padding) {
      }
      OutPutVec.subvec(i, OutPutVec.size()-1) += InputSignalPadded.subvec(0, OutPutVec.size()-1-i);
    }
  }
  if(FiltFilt) {
    OutPutVec = FirFilteringInternal(arma::reverse(OutPutVec), FIR_FILTER, false, BatchSize, false, CORES);
    OutPutVec = arma::reverse(OutPutVec);
    return OutPutVec.subvec(0,size(SIGNAL));
   // return Rcpp::NumericVector(OutPutVec.subvec(0,size(SIGNAL)).begin(),OutPutVec.subvec(0,size(SIGNAL)).end());
  }
  if(!padding) {
    return OutPutVec; 
    //return Rcpp::NumericVector(OutPutVec.begin(),OutPutVec.end());   
  } else {
    return OutPutVec.subvec(prePaddingArea, size(SIGNAL));
    
//    return Rcpp::NumericVector(OutPutVec.subvec(prePaddingArea, size(SIGNAL)).begin(),OutPutVec.subvec(prePaddingArea,size(SIGNAL)).end());
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
  if(SIGNAL.size()>BatchSize) {
    BatchSize = SIGNAL.size();
  }
  return FirFilteringInternal(SIGNAL, FIR_FILTER, FiltFilt, BatchSize, true, CORES);
}