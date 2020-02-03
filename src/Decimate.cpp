#include <RcppArmadillo.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' Decimate
//' 
//' This function downsamples a signal using a preset FIR filter which should be filtering at half the target frequency.
//'
//' @param SIGNAL A numeric vector with length N.
//' @param FIR_FILTER A predefined FIR filter as numeric vector (low-pass filter for <= target frequency).
//' @param M An int representing downsampling factor.
//' @return Returns a numeric vector with length N/M.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector decimate(arma::vec& SIGNAL,
                         const arma::vec& FIR_FILTER,
                         const int& M) {
  arma::colvec PaddedSignal = arma::zeros(SIGNAL.size()+FIR_FILTER.size()*2);
  const arma::uvec& INDEX = arma::conv_to< arma::uvec > ::from(arma::linspace(FIR_FILTER.size(),PaddedSignal.size()-FIR_FILTER.size(), PaddedSignal.size()-FIR_FILTER.size()*2));
  PaddedSignal.elem(INDEX) = SIGNAL;
  const unsigned int OutputLength = std::floor(SIGNAL.size()/M);
  arma::vec DecSignal = arma::zeros(OutputLength);
  const int HalfFilter = FIR_FILTER.size()/2;
  for (unsigned int i=0; i<DecSignal.size(); ++i) {
    DecSignal.at(i) = arma::dot(FIR_FILTER, PaddedSignal.subvec((i*M+HalfFilter), size(FIR_FILTER)));
  } 
  return Rcpp::NumericVector(DecSignal.begin(),DecSignal.end());
}