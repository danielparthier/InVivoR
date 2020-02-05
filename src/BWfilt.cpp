#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]

//' Butterworth filter c++
//' 
//' This function returns a filtered Signal.
//'
//' @param InputSignal A complex matrix from FFTW.
//' @param SamplingFrequency A double indicating sampling frequency.
//' @param ORDER An int as filtering order (default = 2).
//' @param f0 A double as cutoff frequency (default = 10).
//' @param type A string indicating the filter type ("low", "high"). The default is "low".
//' @param CORES An int indicating the number of threads used (default = 1).
//' @return Filtered signal as numeric vector.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector BWFiltCpp(arma::vec& InputSignal,
                         const double& SamplingFrequency,
                         const int& ORDER = 2,
                         const double& f0 = 10,
                         const std::string type = "low",
                         const int& CORES = 1) {
  omp_set_num_threads(CORES);
  int InputLength = std::pow(2, std::ceil(std::log2(InputSignal.size())));
  arma::vec InputSignalPadded = arma::zeros<arma::vec>(InputLength);
  int signalLength = InputSignal.size();
  int startSignal = (InputLength-signalLength)*0.5;
  InputSignalPadded.subvec(startSignal, size(InputSignal)) = InputSignal;
  arma::cx_mat InputFFT = arma::fft(InputSignalPadded);
  int OrderTerm = ORDER*2;
  int SignalLength = InputFFT.size();
  if (f0 > 0) {
    int HalfLength = SignalLength*0.5;
    double BinWidth = SamplingFrequency/SignalLength;
    if(type.compare("low") == 0) {
      BinWidth = BinWidth/f0;
#pragma omp parallel for shared(f0, HalfLength, InputFFT, BinWidth, OrderTerm, SignalLength) schedule(dynamic) default(none)
      for (int i=0; i<HalfLength; ++i) {
        double gain = 1/std::sqrt(1+std::pow((BinWidth * i), OrderTerm));
        InputFFT.at(i) *= gain;
        InputFFT.at(SignalLength-(i)) *= gain;
      } 
    } if(type.compare("high") == 0) {
      BinWidth = f0/BinWidth;
#pragma omp parallel for shared(f0, HalfLength, InputFFT, BinWidth, OrderTerm, SignalLength) schedule(dynamic) default(none)
      for (int i=0; i<HalfLength; ++i) {
        double gain = 1/std::sqrt(1+std::pow(f0/(BinWidth * (i+1)), OrderTerm));
        InputFFT.at(i) *= gain;
        InputFFT.at(SignalLength-(i)) *= gain;
      }
    } if((type.compare("low") != 0) and (type.compare("high") != 0)) {
      Rcpp::stop("No valid type parameter.");
    }
  }
  InputSignalPadded = arma::real(arma::ifft(InputFFT));
  InputSignalPadded = InputSignalPadded.subvec(startSignal, size(InputSignal));
  return Rcpp::NumericVector(InputSignalPadded.begin(),InputSignalPadded.end());
}