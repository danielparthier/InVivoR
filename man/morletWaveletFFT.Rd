% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{morletWaveletFFT}
\alias{morletWaveletFFT}
\title{Morlet wavelet (frequency domain)}
\arguments{
\item{angFreq}{A numeric sequence of angular frequency (0 to 2pi).}

\item{sigma}{A double indicating the shape parameter of the wavelet.}
}
\value{
Morlet wavelet as numeric vector.
}
\description{
This function returns a morlet wavelet in the frequency domain. It can be used for implementation via inverse FFT.
}
