//#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]

//' @title Butterworth filter
//' 
//' @description This function returns a modified FFT.
//'
//' @name BWFilterCpp
//' @param InputFFT A complex matrix from FFTW.
//' @param SamplingFrequency A double indicating sampling frequency.
//' @param ORDER An int as filtering order (default = 2).
//' @param f0 A double as cutoff frequency.
//' @param type A string indicating the filter type ("low", "high"). The default is "low".
//' @param CORES An int indicating the number of threads used (default = 1).
//' 
//' @return Complex armadillo column vector.
//' @export
// [[Rcpp::export]]
arma::cx_vec BWFilterCpp(arma::cx_vec& InputFFT,
                      const double& SamplingFrequency,
                      const int& ORDER = 2,
                      const double& f0 = 10,
                      const std::string type = "low",
                      const int& CORES = 1) {
  omp_set_num_threads(CORES);
  int OrderTerm = ORDER*2;
  int SignalLength = InputFFT.size();
  if (f0 > 0) {
    int HalfLength = SignalLength/2;
    double BinWidth = SamplingFrequency/SignalLength;
    if(type == "low") {
      BinWidth = BinWidth/f0;
#pragma omp parallel for shared(f0, HalfLength, InputFFT, BinWidth, OrderTerm, SignalLength) schedule(dynamic) default(none)
      for (int i=0; i<HalfLength; ++i) {
        double gain = 1/std::sqrt(1+std::pow((BinWidth * i), OrderTerm));
        InputFFT.at(i) *= gain;
        InputFFT.at(SignalLength-(i)) *= gain;
      } 
    } if(type == "high") {
      BinWidth = f0/BinWidth;
#pragma omp parallel for shared(f0, HalfLength, InputFFT, BinWidth, OrderTerm, SignalLength) schedule(dynamic) default(none)
      for (int i=0; i<HalfLength; ++i) {
        double gain = 1/std::sqrt(1+std::pow(f0/(BinWidth * (i+1)), OrderTerm));
        InputFFT.at(i) *= gain;
        InputFFT.at(SignalLength-(i)) *= gain;
      }
    } else {
      Rcpp::stop("No valid type parameter.");
    }
  }
  return InputFFT;
}