#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Decimate
//' 
//' @description This function downsamples a signal using a preset FIR filter which should be filtering at half the target frequency.
//' 
//' @name decimate
//' @param SIGNAL A numeric vector with length N.
//' @param FIR_FILTER A predefined FIR filter as numeric vector (low-pass filter for <= target frequency).
//' @param M An int representing downsampling factor (default = 20).
//' @param CORES An integer indicating number of threads used (default = 1).
//' 
//' @return Returns a numeric vector with length N/M.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector decimate(const arma::vec& SIGNAL,
                             const arma::vec& FIR_FILTER,
                             const int& M = 20,
                             const int CORES = 1) {
  omp_set_num_threads(CORES);
  const long long int SignalLength = SIGNAL.size();
  const long long int OutputLength = std::floor(SignalLength/M);
  arma::vec DecSignal = arma::zeros(OutputLength);
  long long int HalfFilter;
  int halfOff = 0;
  if(FIR_FILTER.size() % 2) {
    HalfFilter = FIR_FILTER.size()/2+1;
    halfOff = 1;
  } else {
    HalfFilter = FIR_FILTER.size()/2;
  }
#pragma omp parallel for shared(FIR_FILTER, M, DecSignal, SIGNAL, HalfFilter, halfOff) schedule(dynamic) // default(none)
  for (long long int i=0; i<OutputLength-1; ++i) {
    if(((i*M-HalfFilter)>=0) & ((i*M+HalfFilter)<SignalLength)) {
      DecSignal.at(i) = arma::dot(FIR_FILTER, SIGNAL.subvec((i*M-HalfFilter), size(FIR_FILTER)));
    } else if((i*M<HalfFilter) & ((i*M+HalfFilter)<SignalLength)) {
      arma::colvec PaddedSignal(size(FIR_FILTER));
      PaddedSignal.fill(SIGNAL.front());
      PaddedSignal.subvec(HalfFilter-i*M-halfOff, PaddedSignal.size()-1) = SIGNAL.subvec(0,HalfFilter+i*M-1);
      DecSignal.at(i) = arma::dot(FIR_FILTER, PaddedSignal);
    } else if((i*M+HalfFilter)>=SignalLength) {
      arma::colvec PaddedSignal(size(FIR_FILTER));
      PaddedSignal.fill(SIGNAL.back());
      PaddedSignal.subvec(0, size(SIGNAL.subvec(i*M-HalfFilter,SIGNAL.size()-1))) = SIGNAL.subvec(i*M-HalfFilter,SIGNAL.size()-1);
      DecSignal.at(i) = arma::dot(FIR_FILTER, PaddedSignal);
    } 
  }
  return Rcpp::NumericVector(DecSignal.begin(),DecSignal.end());
}