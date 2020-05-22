// function to apply FIR filter to data using convolution and zero-padding
#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' FIR filtering
//' 
//' This function applies an FIR filter to a signal an returns the filtered trace.
//'
//' @param SIGNAL A numeric vector.
//' @param FIR_FILTER A numeric vector which can be used as FIR filter.
//' @param FiltFilt A bool indicating if "filtfilt" mode should be used.
//' @param BatchSize An integer indicating the starting batchsize of the trace (chunk size will be optimised for FFT).
//' @param CORES An integer indicating what number of cores should be used.
//' @return Returns numeric vector which is the FIR filtered original signal.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector FirFiltering(const arma::colvec& SIGNAL,
                                        const arma::colvec& FIR_FILTER,
                                        bool FiltFilt = true,
                                        unsigned int BatchSize = 1e4,
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
    Rcpp::stop("ChunkLength too long for signal");
  } else if(ChunkLength < 3*FIR_FILTER.size()) {
    ChunkLength = std::pow(2, std::ceil(std::log2(3*FIR_FILTER.size())));
    BatchSize = ChunkLength/2;
  }
  arma::vec OutPutVec = arma::zeros<arma::vec>(SIGNAL.size());
  int startSignal = (ChunkLength-BatchSize)*0.5;
  arma::vec FilterPadded = arma::zeros<arma::vec>(ChunkLength);
  FilterPadded.subvec(0, size(FIR_FILTER)) = FIR_FILTER;
  Rcpp::Rcout << "1" << std::endl;
  arma::cx_mat Filter_FFT = arma::fft(FilterPadded);
  int BatchJump = BatchSize;
#pragma omp parallel for shared(OutPutVec, SIGNAL,BatchSize, ChunkLength, startSignal, Filter_FFT, BatchJump) schedule(dynamic) default(none)
  for(unsigned long int i = 0; i<SIGNAL.size(); i += BatchJump) { 
    if((i+ChunkLength-startSignal) > SIGNAL.size()) {
      BatchSize = SIGNAL.size()-i;
    }
    arma::vec InputSignalPadded = arma::zeros<arma::vec>(ChunkLength);
    InputSignalPadded.subvec(startSignal, size(SIGNAL.subvec(i, i+BatchSize-1))) = SIGNAL.subvec(i, i+BatchSize-1);
    InputSignalPadded = arma::real(arma::ifft(arma::fft(InputSignalPadded)%Filter_FFT));
#pragma omp critical
    if((i>0) & ((i+ChunkLength+1) < OutPutVec.size())) {
      OutPutVec.subvec(i-startSignal, size(InputSignalPadded)) += InputSignalPadded;
    } else if(i == 0) {
      OutPutVec.subvec(0, (InputSignalPadded.size()-startSignal-1)) += InputSignalPadded.subvec(startSignal, InputSignalPadded.size()-1);//InputSignalPadded.tail(InputSignalPadded.size()-startSignal);
    } else if((i+ChunkLength+1)>OutPutVec.size()) {
      
      OutPutVec.subvec(i-startSignal, OutPutVec.size()-1) += InputSignalPadded.subvec(0, size(OutPutVec.subvec(i-startSignal, OutPutVec.size()-1)));//startSignal+BatchSize-1);//InputSignalPadded.size()-startSignal-1);
    }
  }
  if(FiltFilt) {
    OutPutVec = FirFilteringOverlap(arma::reverse(OutPutVec), FIR_FILTER, false, BatchSize, CORES);
    OutPutVec = arma::reverse(OutPutVec);
    return Rcpp::NumericVector(OutPutVec.begin(),OutPutVec.end());
  }
  return Rcpp::NumericVector(OutPutVec.begin(),OutPutVec.end());
}
