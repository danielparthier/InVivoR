#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' Morlet wavelet (time domain)
//' 
//' This function returns a complex morlet wavelet in the time domain. It can be 
//' used in convolution.
//'
//' @param t A numeric sequence of time (-t/2 to t/2) with steps of sampling frequency.
//' @param sigma A double indicating the shape parameter of the wavelet.
//' @return Morlet wavelet as complex vector.
//' @export
// [[Rcpp::export]]
arma::cx_vec morletWavlet(arma::vec& t,
                          const double& sigma) {
  double pi4 = std::pow(arma::datum::pi, -0.25);
  double c0 = std::pow((1+std::exp(-sigma*sigma)-2*std::exp(-0.75*sigma*sigma)), -0.5);
  double k0 = std::exp(-0.5*sigma*sigma);
  arma::vec sigmaVec(t.n_elem);
  sigmaVec.fill(sigma);
  sigmaVec %=t;
  arma::vec tSq = arma::vec(t.memptr(), t.n_elem, true, false);
  tSq %= -t*0.5;
  arma::cx_vec WaveletOut(t.n_elem, arma::fill::zeros);
  WaveletOut.set_imag(sigmaVec);
  return c0*pi4*arma::exp(WaveletOut)%arma::exp(tSq-k0);
  }



//' Morlet wavelet (frequency domain)
//' 
//' This function returns a morlet wavelet in the frequency domain. It can be 
//' used for implementation via inverse FFT.
//'
//' @param angFreq A numeric sequence of angular frequency (0 to 2pi).
//' @param sigma A double indicating the shape parameter of the wavelet.
//' @return Morlet wavelet as numeric vector.
//' @export
// [[Rcpp::export]]
arma::vec morletWaveletFFT(const arma::vec& angFreq,
                           double& sigma) {
  double pi4 = std::pow(arma::datum::pi, -0.25);
  double c0 = std::pow((1+std::exp(-sigma*sigma)-2*std::exp(-0.75*sigma*sigma)), -0.5);
  double k0 = std::exp(-0.5*sigma*sigma);
  return c0*pi4*(arma::exp(-0.5*(angFreq-sigma)%(angFreq-sigma))-(-k0*arma::exp(-0.5*angFreq%angFreq)));
}


//' Application of morlet wavelet (frequency domain)
//' 
//' This function returns the convolution of a complex morlet daughter wavelet with the signal.
//' 
//' @param SignalFFT A complex vector of the signal FFT.
//' @param scale A double indicating the scale parameter of daughter wavelet.
//' @param morletFFT A vector of wavelet in frequency domain.
//' @param LNorm A double indicating the L normalisation (power of 1/LNorm, default = 2).
//' @return Morlet wavelet as numeric vector.
//' @export
// [[Rcpp::export]]
arma::cx_vec morletWT(const arma::cx_vec& SignalFFT,
                      const double& scale, arma::vec morletFFT,
                      const double& LNorm = 2) {
  return arma::ifft(std::pow(scale * arma::datum::pi*2, 1/LNorm) * SignalFFT % arma::conj(morletFFT));
}


//' Wavelet transform
//' 
//' This function performs a wavelet transform of a signal for different scales 
//' and returns a complex matrix from the convolution with a complex wavelet in 
//' the frequency domain.
//' 
//' @param Signal A numeric vector.
//' @param frequencies A vector indicating the frequencies which should be analysed.
//' @param samplingfrequency A double indicating the sampling frequency in Hz.
//' @param sigma A double indicating the shape parameter of the wavelet.
//' @param LNorm A double indicating the L normalisation (power of 1/LNorm, default = 2).
//' @param CORES An integer indicating number of threads used (default = 1). 
//' @return Wavelet transform as complex matrix.
//' 
//' @examples # test signal
//' testSignal <- sin(seq(0,32*pi, length.out = 4000))*6
//' testSignal <- testSignal+sin(seq(0,84*pi, length.out = 4000))*10
//' 
//' # apply wavelet tranform
//' WTmat <- WT(Signal = testSignal, frequencies = seq(from = 0.2,to = 20, by = 0.1),
//' samplingfrequency = 1e3, sigma = 12, LNorm = 2, CORES = 1)
//' 
//' # plot real part of WT
//' image(x = Re(WTmat), col = hcl.colors(n = 1000, palette = "viridis"), useRaster = T)
//' 
//' # plot power of WT
//' image(x = abs(WTmat)^2, col = hcl.colors(n = 1000, palette = "viridis"), useRaster = T)
//' 
//' # plot phase
//' image(x = atan2(y = Im(WTmat), x = Re(WTmat)), col = hcl.colors(n = 1000, palette = "viridis"), useRaster = T)
//' 
//' @export
// [[Rcpp::export]]
arma::cx_mat WT(const arma::vec& Signal,
                const arma::vec& frequencies,
                const double& samplingfrequency,
                double& sigma,
                const double& LNorm = 2,
                int CORES = 1) {
  int scaleLength = frequencies.n_elem;
  int SignalLength = Signal.n_elem;
  omp_set_num_threads(CORES);
  arma::vec SignalPadded = arma::zeros<arma::vec>(SignalLength*3);
  SignalPadded.subvec(SignalLength+1,size(Signal)) = Signal;
  arma::vec scale = samplingfrequency/arma::vec(frequencies.memptr(), scaleLength)/((4*arma::datum::pi)/(sigma+std::sqrt(2+sigma*sigma)));
  arma::vec angFreq = arma::linspace<arma::vec>(0, 2*arma::datum::pi, SignalPadded.n_elem);
  arma::cx_vec SignalFFT = arma::fft(SignalPadded);
  arma::cx_mat WTMat = arma::zeros<arma::cx_mat>(SignalLength, scaleLength);
#pragma omp parallel for shared(SignalFFT, angFreq, scale, scaleLength, sigma, LNorm) schedule(static) 
  for(int i = 0; i < scaleLength; ++i) {
    arma::cx_vec tmp =  morletWT(SignalFFT, scale.at(i), morletWaveletFFT(angFreq*scale.at(i), sigma), LNorm);
    WTMat.unsafe_col(i) = tmp.subvec(SignalLength+1,SignalLength*2);
  }
  return WTMat;
}

//' Wavelet transform (from ERP matrix)
//' 
//' This function performs a wavelet transform of a signal for different scales 
//' and returns a complex matrix from the convolution with a complex wavelet in 
//' the frequency domain.
//' 
//' @param ERPMat A numeric matrix with rows for ERP signals and columns as time domain.
//' @param frequencies A vector indicating the frequencies which should be analysed.
//' @param samplingfrequency A double indicating the sampling frequency in Hz.
//' @param sigma A double indicating the shape parameter of the wavelet.
//' @param LNorm A double indicating the L normalisation (power of 1/LNorm, default = 2).
//' @param CORES An integer indicating number of threads used (default = 1). 
//' @return Wavelet transform as complex cube (each slice is from one ERP).
//' 
//' @examples # Generate test signal
//' testSignal <- sin(seq(0,32*pi, length.out = 4000))*6
//' testSignal <- testSignal+sin(seq(0,84*pi, length.out = 4000))*10
//' 
//' # Generate ERP matrix
//' ERPmat <- ERPMat(Trace = testSignal, Onset = (1:10)*200, End = (1:10)*400)
//'   
//' # Apply WT to all ERPs
//' WTCube <- WTbatch(ERPMat = ERPmat, frequencies = seq(0.2,20, 0.2), samplingfrequency = 1000, sigma = 6, LNorm = 2, CORES = 1)
//'     
//' # Cube dimensions
//' dim(WTCube)
//'       
//' # Real part of wavelet transform for different ERPs
//' image(x = Re(WTCube[,,1]), col = hcl.colors(n = 1000, palette = "viridis"), useRaster = T)
//' image(x = Re(WTCube[,,5]), col = hcl.colors(n = 1000, palette = "viridis"), useRaster = T)
//' image(x = Re(WTCube[,,10]), col = hcl.colors(n = 1000, palette = "viridis"), useRaster = T)
//' 
//' @export
// [[Rcpp::export]]
arma::cx_cube WTbatch(arma::mat& ERPMat,
                      const arma::vec& frequencies,
                      const double& samplingfrequency,
                      double& sigma,
                      const double& LNorm = 2,
                      const int& CORES = 1) {
  int scaleLength = frequencies.n_elem;
  int SignalLength = ERPMat.n_cols;
  int SignalCount = ERPMat.n_rows;
  inplace_trans(ERPMat);
  omp_set_num_threads(CORES);
  arma::vec scale = samplingfrequency/arma::vec(frequencies.memptr(), scaleLength)/((4*arma::datum::pi)/(sigma+std::sqrt(2+sigma*sigma)));
  arma::vec angFreq = arma::linspace<arma::vec>(0, 2*arma::datum::pi, SignalLength*3);
  arma::cx_cube WTCube = arma::zeros<arma::cx_cube>(SignalLength,scaleLength, SignalCount);
  arma::mat ScaleMat = arma::mat(SignalLength*3,scaleLength,arma::fill::zeros);
  for(int i = 0; i < scaleLength; ++i) {
    ScaleMat.unsafe_col(i) = morletWaveletFFT(angFreq*scale.at(i), sigma);
  }
#pragma omp parallel for shared(ERPMat, ScaleMat, scale, scaleLength, SignalCount, SignalLength, sigma, LNorm) schedule(static) 
  for(int j = 0; j < SignalCount; ++j) {
    arma::cx_mat WTMat = arma::zeros<arma::cx_mat>(SignalLength, scaleLength);
    arma::vec SignalPadded = arma::zeros<arma::vec>(SignalLength*3);
    SignalPadded.subvec(SignalLength+1,size(ERPMat.col(j))) = ERPMat.col(j);
    arma::cx_vec SignalFFT = arma::fft(SignalPadded);
    for(int k = 0; k < scaleLength; ++k) {
      arma::cx_vec tmp =  morletWT(SignalFFT, scale.at(k), ScaleMat.unsafe_col(k), LNorm);
      WTMat.unsafe_col(k) = tmp.subvec(SignalLength+1,SignalLength*2);
    }
    WTCube.slice(j) = WTMat;
  }
  return WTCube;
}

//' Wavelet power matrix (from wavelet power cube)
//' 
//' This function computes the average power of several WT which can be returned as raw power
//' or as z-score. If the z-score is returned then every slice of the cube is z-transformed 
//' independently and the average is calculated in z-direction.
//' 
//' @param x A cube with power matrices each slice representing ERP.
//' @return An average power matrix (raw or as z-score).
//' @export
// [[Rcpp::export]]
arma::mat PowerMat(const arma::cube& x,
                   const bool& ZScore = false) {
  int DIM_X = x.n_rows;
  int DIM_Y = x.n_cols;
  int DIM_Z = x.n_slices;
  long double mean;
  long double sd;
  arma::cube Cube = arma::cube(x.memptr(), DIM_X, DIM_Y, DIM_Z);
  if(ZScore) {
    for(int i = 0; i < DIM_Z; ++i) {
      mean = arma::mean(arma::vectorise(Cube.slice(i)));
      sd = arma::stddev(arma::vectorise(Cube.slice(i)));
      Cube.slice(i) -= mean;
      Cube.slice(i) /= sd;
    }
  }
  return arma::mean(Cube, 2);
}