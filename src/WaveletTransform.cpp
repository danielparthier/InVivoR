//#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Morlet wavelet (time domain)
//' 
//' This function returns a complex morlet wavelet in the time domain. It can be 
//' used for convolution.
//' 
//' @name morletWavlet
//' @param t A numeric sequence of time (-t/2 to t/2) with steps of sampling frequency.
//' @param sigma A double indicating the shape parameter of the wavelet.
//' 
//' @return Morlet wavelet as complex vector.
arma::cx_vec morletWavlet(arma::vec& t,
                          const double& sigma) {
  const double pi4 = std::pow(arma::datum::pi, -0.25);
  const double c0 = std::pow((1+std::exp(-sigma*sigma)-2*std::exp(-0.75*sigma*sigma)), -0.5);
  const double k0 = std::exp(-0.5*sigma*sigma);
  arma::vec sigmaVec(t.n_elem);
  sigmaVec.fill(sigma);
  sigmaVec %= t;
  arma::vec tSq = arma::vec(t.memptr(), t.n_elem, true, false);
  tSq %= -t*0.5;
  arma::cx_vec WaveletOut(t.n_elem, arma::fill::zeros);
  WaveletOut.set_imag(sigmaVec);
  return c0*pi4*arma::exp(WaveletOut)%arma::exp(tSq-k0);
}



//' @title Morlet wavelet (frequency domain)
//' 
//' @description This function returns a morlet wavelet in the frequency domain. It can be used for implementation via inverse FFT.
//'
//' @name morletWaveletFFT
//' @param angFreq A numeric sequence of angular frequency (0 to 2pi).
//' @param sigma A double indicating the shape parameter of the wavelet.
//' 
//' @return Morlet wavelet as numeric vector.
arma::vec morletWaveletFFT(const arma::vec& angFreq,
                           const double& sigma) {
  double pi4 = std::pow(arma::datum::pi, -0.25);
  double c0 = std::pow((1+std::exp(-sigma*sigma)-2*std::exp(-0.75*sigma*sigma)), -0.5);
  double k0 = std::exp(-0.5*sigma*sigma);
  return c0*pi4*(arma::exp(-0.5*(angFreq-sigma)%(angFreq-sigma))-(-k0*arma::exp(-0.5*angFreq%angFreq)));
}


//' @title Application of morlet wavelet (frequency domain)
//'
//' @description This function returns the convolution of a complex morlet daughter wavelet with the signal.
//' 
//' @name morletWT
//' @param SignalFFT A complex vector of the signal FFT.
//' @param scale A double indicating the scale parameter of daughter wavelet.
//' @param morletFFT A vector of wavelet in frequency domain.
//' @param LNorm A double indicating the L normalisation (power of 1/LNorm, default = 2).
//' 
//' @return Morlet wavelet as numeric vector.
arma::cx_vec morletWT(const arma::cx_vec& SignalFFT,
                      const double& scale, arma::vec morletFFT,
                      const double& LNorm = 2) {
  return arma::ifft(std::pow(scale * arma::datum::pi*2, 1/LNorm) * SignalFFT % arma::conj(morletFFT));
}


//' @title Wavelet transform
//' 
//' This function performs a wavelet transform of a signal for different scales 
//' and returns a complex matrix from the convolution with a complex wavelet in 
//' the frequency domain.
//' 
//' @name WT
//' @param Signal A numeric vector.
//' @param frequencies A vector indicating the frequencies which should be analysed.
//' @param SamplingRate A double indicating the sampling frequency in Hz (default = 1000).
//' @param sigma A double indicating the shape parameter of the wavelet (default = 6).
//' @param LNorm A double indicating the L normalisation (power of 1/LNorm, default = 2).
//' @param CORES An integer indicating number of threads used (default = 1). 
//' 
//' @return Wavelet transform as complex matrix.
//' @export
// [[Rcpp::export]]
arma::cx_mat WT(const arma::vec& Signal,
                const arma::vec& frequencies,
                const double& SamplingRate = 1e3,
                const double& sigma = 6,
                const double& LNorm = 2,
                int CORES = 1) {
  const int scaleLength = frequencies.n_elem;
  const int SignalLength = Signal.n_elem;
  omp_set_num_threads(CORES);
  //newly added padding
  int paddingLength = std::pow(2, std::ceil(std::log2(SignalLength*2)));
  int SignalStart = std::floor(paddingLength/2);
  arma::vec SignalPadded = arma::zeros<arma::vec>(paddingLength);
  SignalPadded.subvec(SignalStart,size(Signal)) = Signal;
  arma::vec scale = SamplingRate/arma::vec(frequencies.memptr(), scaleLength)/((4*arma::datum::pi)/(sigma+std::sqrt(2+sigma*sigma)));
  arma::vec angFreq = arma::linspace<arma::vec>(0, 2*arma::datum::pi, SignalPadded.n_elem);
  arma::cx_vec SignalFFT = arma::fft(SignalPadded);
  arma::cx_mat WTMat = arma::zeros<arma::cx_mat>(SignalLength, scaleLength);
#pragma omp parallel for shared(SignalFFT, angFreq, scale, sigma, LNorm) schedule(static) 
  for(int i = 0; i < scaleLength; ++i) {
    arma::cx_vec tmp =  morletWT(SignalFFT, scale.at(i), morletWaveletFFT(angFreq*scale.at(i), sigma), LNorm);
    WTMat.unsafe_col(i) = tmp.subvec(SignalStart,SignalStart+SignalLength-1);
  }
  return WTMat;
}

//' @title Wavelet transform (from ERP matrix)
//' 
//' This function performs a wavelet transform of a signal for different scales 
//' and returns a complex matrix from the convolution with a complex wavelet in 
//' the frequency domain.
//' 
//' @name WTbatch
//' @param ERPMat A numeric matrix with rows for ERP signals and columns as time domain.
//' @param frequencies A vector indicating the frequencies which should be analysed.
//' @param SamplingRate A double indicating the sampling frequency in Hz (default = 1000).
//' @param sigma A double indicating the shape parameter of the wavelet (default = 6).
//' @param LNorm A double indicating the L normalisation (power of 1/LNorm, default = 2).
//' @param CORES An integer indicating number of threads used (default = 1).
//' @param compression An integer indicating number of threads used (default = 1).
//' @param PhaseAnalysis An integer indicating number of threads used (default = 1) 
//' 
//' @return List with wavelet transform as complex cube (each slice is from one ERP or when compressed one matrix), 
//' rho vector length and mean phase.
//' @export
// [[Rcpp::export]]
Rcpp::List WTbatch(arma::mat ERPMat,
                   const arma::vec& frequencies,
                   const double& SamplingRate = 1e3,
                   const double& sigma = 6,
                   const double& LNorm = 2,
                   int CORES = 1L,
                   bool compression = false,
                   bool PhaseAnalysis = false) {
  const int scaleLength = frequencies.n_elem;
  const int SignalLength = ERPMat.n_cols;
  const int SignalCount = ERPMat.n_rows;
  if(CORES>SignalCount) {
    CORES = SignalCount;
  }
  arma::cx_cube WTCube;
  arma::mat WTCubeCos;
  arma::mat WTCubeSin;
  inplace_trans(ERPMat);
  omp_set_num_threads(CORES);
  arma::vec scale = SamplingRate/arma::vec(frequencies.memptr(), scaleLength)/((4*arma::datum::pi)/(sigma+std::sqrt(2+sigma*sigma)));
  int paddingLength = std::pow(2, std::ceil(std::log2(SignalLength*2)));
  int SignalStart = std::floor(paddingLength/2);
  arma::vec angFreq = arma::linspace<arma::vec>(0, 2*arma::datum::pi, paddingLength);
  
  if(PhaseAnalysis) {
    if(SignalCount == 1) {
      Rcpp::warning("Only one ERP - no phase analysis possible");
      PhaseAnalysis = false;
    } else {
      WTCubeCos = arma::zeros<arma::mat>(SignalLength, scaleLength);
      WTCubeSin = arma::zeros<arma::mat>(SignalLength, scaleLength);  
    }
  }
  if(compression) {
    WTCube = arma::zeros<arma::cx_cube>(SignalLength,scaleLength, 1);
  } else {
    WTCube = arma::zeros<arma::cx_cube>(SignalLength,scaleLength, SignalCount);
  }
  arma::mat ScaleMat = arma::mat(paddingLength,scaleLength,arma::fill::zeros);
  for(int i = 0; i < scaleLength; ++i) {
    ScaleMat.unsafe_col(i) = morletWaveletFFT(angFreq*scale.at(i), sigma);
  }
#pragma omp parallel for shared(ERPMat, ScaleMat, scale, sigma, LNorm) schedule(static) 
  for(int j = 0; j < SignalCount; ++j) {
    arma::cx_mat WTMat = arma::zeros<arma::cx_mat>(SignalLength, scaleLength);
    arma::vec SignalPadded = arma::zeros<arma::vec>(paddingLength);
    SignalPadded.subvec(SignalStart,size(ERPMat.col(j))) = ERPMat.col(j);
    arma::cx_vec SignalFFT = arma::fft(SignalPadded);
    for(int k = 0; k < scaleLength; ++k) {
      arma::cx_vec tmp =  morletWT(SignalFFT, scale.at(k), ScaleMat.col(k), LNorm);
      WTMat.unsafe_col(k) = tmp.subvec(SignalStart, size(ERPMat.col(j)));
    }
#pragma omp critical
    if(PhaseAnalysis) {
      arma::mat PhaseMatTmp = atan2(arma::imag(WTMat), arma::real(WTMat));
      WTCubeCos += arma::cos(PhaseMatTmp)/(SignalCount-1);
      WTCubeSin += arma::sin(PhaseMatTmp)/(SignalCount-1);
    }
#pragma omp critical
    if(compression) {
      WTCube.slice(0) += WTMat/(SignalCount-1);
    } else if(compression == false) {
      WTCube.slice(j) = WTMat;
    }
  }
  Rcpp::Rcout << "finished ERP" << std::endl;
  if(PhaseAnalysis) {
    if(compression) {
      return Rcpp::List::create(Rcpp::Named("Raw") = WTCube.slice(0),
                                Rcpp::Named("Rho") = arma::sqrt(WTCubeSin%WTCubeSin+WTCubeCos%WTCubeCos),
                                Rcpp::Named("MeanPhase") = arma::atan2(WTCubeSin, WTCubeCos),
                                Rcpp::Named("Frequencies") = Rcpp::NumericVector(frequencies.begin(),frequencies.end()));
    } else {
      return Rcpp::List::create(Rcpp::Named("Raw") = WTCube,
                                Rcpp::Named("Rho") = arma::sqrt(WTCubeSin%WTCubeSin+WTCubeCos%WTCubeCos),
                                Rcpp::Named("MeanPhase") = arma::atan2(WTCubeSin, WTCubeCos),
                                Rcpp::Named("Frequencies") = Rcpp::NumericVector(frequencies.begin(),frequencies.end()));
    }
    
  } else {
    if(compression) {
      return Rcpp::List::create(Rcpp::Named("Raw") = WTCube.slice(0),
                                Rcpp::Named("Frequencies") = Rcpp::NumericVector(frequencies.begin(),frequencies.end()));
    } else {
      return Rcpp::List::create(Rcpp::Named("Raw") = WTCube,
                                Rcpp::Named("Frequencies") = Rcpp::NumericVector(frequencies.begin(),frequencies.end()));
    }
  }
}

//' @title Wavelet power matrix (from wavelet power cube)
//' 
//' This function computes the average power of several WT which can be returned as raw power
//' or as z-score. If the z-score is returned then every slice of the cube is z-transformed 
//' independently and the average is calculated in z-direction.
//' 
//' @name PowerMat
//' @param x A cube with power matrices each slice representing ERP.
//' @param ZScore A bool indicating if Z-score should be computed.
//' 
//' @return An average power matrix (raw or as z-score).
//' @export
// [[Rcpp::export]]
arma::mat PowerMat(const arma::cube& x,
                   const bool& ZScore = false) {
  const int DIM_X = x.n_rows;
  const int DIM_Y = x.n_cols;
  const int DIM_Z = x.n_slices;
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

//' @title Synchrosqueezed wavelet power matrix (from wavelet power matrix)
//' 
//' This function computes the synchrosqueezed wavelet transform as proposed by Daubechies and Maes (1996). Wavelet 
//' coefficients of the wavelet will be reassigned according to the instantaneous frequency in the transform.
//' 
//' @name WTSqueeze
//' @param WT A complex matrix representing the wavelet transform.
//' @param frequencies A vector indicating the frequencies which should be analysed.
//' @param SamplingRate A double indicating the sampling frequency in Hz (default = 1000).
//' @param sigma A double indicating the shape parameter of the wavelet (default = 6).
//' @param CORES An integer indicating number of threads used (default = 1). 
//' 
//' @return A complex matrix representing the synchrosqueezed wavelet transform.
//' 
//' @export
// [[Rcpp::export]]
arma::cx_mat WTSqueeze(const arma::cx_mat& WT,
                       const arma::vec& frequencies,
                       const double& SamplingRate = 1e3,
                       double sigma = 6,
                       const int& CORES = 1) {
  const int DIM_X =  WT.n_cols;
  const int DIM_Y =  WT.n_rows;
  omp_set_num_threads(CORES);
  
  arma::cx_mat x = arma::cx_mat(WT.memptr(), DIM_Y, DIM_X);
  
  // scale calculation, delta and useful transformations
  arma::vec scale = SamplingRate/frequencies/((4*arma::datum::pi)/(sigma+std::sqrt(2+sigma*sigma)));
  arma::vec scaleDelta = arma::zeros<arma::vec>(scale.n_elem+1);
  scaleDelta.at(0) = frequencies.at(0)+(frequencies.at(0)-frequencies.at(1));
  scaleDelta.subvec(1,size(scale)) = frequencies;
  if(scaleDelta.at(0)==0) {
    scaleDelta.at(0) = scaleDelta.at(1)*2;
  }
  scaleDelta = SamplingRate/scaleDelta;
  scaleDelta /= (4*arma::datum::pi)/(sigma+std::sqrt(2+sigma*sigma));
  scaleDelta = arma::abs(arma::diff(scaleDelta));
  arma::vec scalePower = arma::pow(scale, -1.5);
  scalePower %= scaleDelta;
  
  // angular frequency and delta
  arma::vec wFreq = frequencies*2*arma::datum::pi;
  arma::vec wDelta = arma::zeros<arma::vec>(wFreq.n_elem+1);
  wDelta.at(0) = wFreq.at(0)-(wFreq.at(1)-wFreq.at(0));
  wDelta.subvec(1,size(wFreq)) = wFreq;
  wDelta = arma::abs(arma::diff(wDelta));
  arma::vec wDeltaHalf = wDelta/2;
  arma::vec wDeltaInv = 1/wDelta;
  
  //wavelet differential and instantaneous frequency
  arma::cx_mat wab = arma::cx_mat(x.n_rows, x.n_cols, arma::fill::zeros);
  wab.fill(arma::cx_double(0,-1));
  wab.col(0) = arma::zeros<arma::cx_vec>(x.n_rows);
  wab.submat(1, 0, x.n_rows-1, x.n_cols-1) %= arma::diff(x, 1);
  arma::mat wabAbs = arma::abs(wab / x * SamplingRate);
  arma::inplace_strans(wabAbs);
  arma::inplace_strans(x);
  
  // allocating space for output matrix
  arma::cx_mat Ts = arma::cx_mat(DIM_Y, DIM_X, arma::fill::zeros);
#pragma omp parallel for shared(x, Ts, wabAbs, wFreq, wDeltaHalf, wDeltaInv, scalePower) schedule(static)
  for(int b = 0; b < DIM_Y; ++b) {
    for(int w = 0; w < DIM_X; ++w) {
      // find instantaneous frequencies in wl range
      arma::ucolvec ColK = arma::find(arma::abs(wabAbs.col(b)-wFreq.at(w)) < wDeltaHalf.at(w));
      if(ColK.n_elem > 0) {
        // transform amplitude from a,b space to b,w space
        arma::cx_vec tmp = x.col(b);
        Ts.at(b,w) = wDeltaInv.at(w) * arma::sum(tmp.elem(ColK)) * scalePower.at(w);
      }
    }
  }
  return Ts;
}

//' @title Average complex matrix (from wavelet power cube)
//' 
//' @description This function computes the average complex matrix of a complex cube.
//' 
//' @name CxCubeCollapse
//' @param x A cube with complex matrices each slice representing ERP.
//' 
//' @return An average complex matrix.
//' 
//' @export
// [[Rcpp::export]]
arma::cx_mat CxCubeCollapse(const arma::cx_cube& x) {
  return arma::mean(x, 2);
}

//' @title Coherence/Coherency/Phase Difference function
//' 
//' @description This function computes the coherence, coherency, and the phase difference between two wavelet transforms.
//' 
//' @name WTCoherence
//' @param WT1 A complex matrix representing the wavelet transform.
//' @param WT2 A complex matrix representing the wavelet transform.
//' @param frequencies A vector indicating the frequencies which should be analysed.
//' @param SamplingRate A double indicating the sampling frequency in Hz (default = 1000).
//' @param tKernelWidth A double indicating the sd as smoothing factor in the time domain (default = 0.01).
//' @param sKernelWidth A double indicating the smoothing factor in the scale domain (default = 0.6).
//' 
//' @return A list containing the coherency, coherence, and phase difference of the two wavelet transforms.
//' @export
// [[Rcpp::export]]
Rcpp::List WTCoherence(arma::cx_mat& WT1,
                       arma::cx_mat& WT2,
                       arma::vec& frequencies,
                       const double& SamplingRate = 1e3,
                       const double& tKernelWidth = 0.01,
                       const double& sKernelWidth = 0.6) {
  arma::vec tKernel = arma::normpdf(arma::regspace(0,1/SamplingRate,WT1.n_rows/SamplingRate), 0, tKernelWidth);
  arma::vec sKernel = arma::zeros<arma::vec>(frequencies.n_elem);
  for(unsigned int i = 0; i < frequencies.n_elem; ++i) {
    sKernel.at(i) = std::pow(sKernelWidth, frequencies.at(i));
  }
  int paddingCol = std::pow(2, std::ceil(std::log2(WT1.n_cols*2)));
  int paddingRow = std::pow(2, std::ceil(std::log2(WT1.n_rows*2)));
  arma::mat kernel2D = tKernel*sKernel.t();
  kernel2D /= arma::accu(kernel2D);
  arma::mat kernelPad = arma::zeros<arma::mat>(paddingRow, paddingCol);
  arma::mat WT1Pad = arma::zeros<arma::mat>(paddingRow, paddingCol);
  arma::mat WT2Pad = arma::zeros<arma::mat>(paddingRow, paddingCol);
  arma::cx_mat WT12Pad = arma::zeros<arma::cx_mat>(paddingRow, paddingCol);
  kernelPad(0, 0, size(kernel2D)) = kernel2D;
  WT1Pad(0, 0, size(WT1)) = arma::real(arma::abs(WT1)%arma::abs(WT1));
  WT2Pad(0, 0, size(WT2)) = arma::real(arma::abs(WT2)%arma::abs(WT2));
  WT12Pad(0, 0, size(WT1)) = WT1%arma::conj(WT2);
  arma::cx_mat kernelFFT = arma::fft2(kernelPad);
  arma::cx_mat WT1s = arma::ifft2(kernelFFT%arma::fft2(WT1Pad));
  arma::cx_mat WT2s = arma::ifft2(kernelFFT%arma::fft2(WT2Pad));
  arma::cx_mat WT12s = arma::ifft2(kernelFFT%arma::fft2(WT12Pad));
  WT1s = WT1s(0, 0, size(WT1));
  WT2s = WT2s(0, 0, size(WT2));
  WT12s = WT12s(0, 0, size(WT1));
  return Rcpp::List::create(Rcpp::Named("Coherency") = arma::real(WT12s/arma::sqrt(WT1s%WT2s)),
                            Rcpp::Named("Coherence") = arma::real((WT12s%arma::real(WT12s)+arma::imag(WT12s)%arma::imag(WT12s))/(WT1s%WT2s)),
                            Rcpp::Named("PhaseDiff") = arma::atan(arma::real(WT12s)/arma::imag(WT12s)));
}