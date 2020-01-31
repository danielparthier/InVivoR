#' Butterworth filter
#'
#' This function filters the data based on the butterworth filter. It uses an FFTW implementation to accelerate the filtering process.
#' @param Signal A Numeric Vector as input
#' @param order An Integer indicating the order of filtering (default = 2)
#' @param SamplingFrequency An Integer as sampling frequency of signal (default = 2e4)
#' @param f0 The input for the frequency cutoff
#' @param cores Multicore usages can be activated by setting number of threads (default = 1).
#' @param type Type of filtering can be set as "low", "high" for low-pass and high-pass filtering.
#' @param PaddingType Padding type can be set by as "zero" (zero-padding), "edge" (values of edge are used for padding), "none" (no padding). The default value is set "zero"
#'
#' @return
#' @export
#'
#' @examples
bwFilter <- function(Signal,
                     order = 2,
                     SamplingFrequency = 2e4,
                     f0,
                     cores = 1,
                     type = "zero",
                     PaddingType) {
  PaddingLength <- floor(1/f0*SamplingFrequency)
  if(PaddingType=="zero") {
    FFTMatrix <- t(fftw::FFT(c(rep(0,PaddingLength),Signal,rep(0, PaddingLength))))
  } else if(PaddingType=="edge") {
    FFTMatrix <- t(fftw::FFT(c(rep(Signal[1],PaddingLength),Signal,rep(Signal[length(Signal)], PaddingLength))))
  } else if(PaddingType=="none") {
    FFTMatrix <- t(fftw::FFT(Signal))
    PaddingLength <- 0
  }
  
  return(Re(fftw::IFFT(BWFilter(InputFFT = FFTMatrix,
                                SamplingFrequency = SamplingFrequency,
                                ORDER = order,
                                f0 = f0,
                                CORES = cores,
                                type = type)))[(PaddingLength+1):(length(Signal)+PaddingLength)])
}


convCent <- function(x, y) {
  Lx <- length(x)
  Ly <- length(y)
  ConvLength <- 2^ceiling(log2(Lx+Ly-1))
  FFTPlan <- fftw::planFFT(n = ConvLength)
  return(Re(fftw::IFFT(fftw::FFT(x = c(x, rep(0, ConvLength-Lx)), plan = FFTPlan)*fftw::FFT(x = c(y, rep(0, ConvLength-Ly)), plan = FFTPlan)))[(Ly/2+1):(Lx+(Ly/2))])
}

convFilter <- function(x,
                 y,
                 type = "filter") {
  Lx <- length(x)
  Ly <- length(y)
  if (type=="open") {
    return(convCent(x = c(rep(0, Ly), x, rep(0, Ly)), y = y)[(Ly+1):(Ly+Lx)])
  } else if (type=="filter") {
    return(rev(convCent(x = c(rep(0, Ly), rev(convCent(x = c(rep(0, Ly), x, rep(0, Ly)), y = y)[(Ly+1):(Ly+Lx)]), rep(0, Ly)), y = y)[(Ly+1):(Ly+Lx)]))
  }
}

deconv <- function(x, y) {
  Lx <- length(x)
  Ly <- length(y)
  ConvLength <- 2^ceiling(log2(Lx+Ly-1))
  FFTPlan <- fftw::planFFT(n = ConvLength)
  return(Re(fftw::IFFT(fftw::FFT(x = c(x, rep(0, ConvLength-Lx)), plan = FFTPlan)/fftw::FFT(x = c(y, rep(0, ConvLength-Ly)), plan = FFTPlan)))[(Ly/2+1):(Lx+(Ly/2))])
}
