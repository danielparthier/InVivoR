// function to apply FIR filter to data using convolution and zero-padding

#include <RcppArmadillo.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]

//' FIR filtering
//' 
//' This function applies an FIR filter to a signal an returns the filtered trace.
//'
//' @param SIGNAL A numeric vector.
//' @param FIR_FILTER A numeric vector which can be used as FIR filter.
//' @return Returns numeric vector which is the FIR filtered original signal.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector FirFiltering(const arma::colvec& SIGNAL,
                                  const arma::colvec& FIR_FILTER) {
  arma::colvec PaddedSignal = arma::zeros(SIGNAL.size()+FIR_FILTER.size()*2);
  const arma::uvec& INDEX = arma::conv_to< arma::uvec > ::from(arma::linspace(FIR_FILTER.size(),PaddedSignal.size()-FIR_FILTER.size(), PaddedSignal.size()-FIR_FILTER.size()*2));
  PaddedSignal.elem(INDEX) = SIGNAL;
  arma::colvec OutSignal = arma::conv(PaddedSignal, FIR_FILTER, "same");
  OutSignal = arma::reverse(arma::conv(arma::reverse(OutSignal), FIR_FILTER, "same"));
  OutSignal = OutSignal.elem(INDEX);
return Rcpp::NumericVector(OutSignal.begin(),OutSignal.end());
}