% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{FirFiltering}
\alias{FirFiltering}
\title{FIR filtering}
\usage{
FirFiltering(
  SIGNAL,
  FIR_FILTER,
  FiltFilt = TRUE,
  BatchSize = 10000L,
  CORES = 1L
)
}
\arguments{
\item{SIGNAL}{A numeric vector.}

\item{FIR_FILTER}{A numeric vector which can be used as FIR filter.}

\item{FiltFilt}{A bool indicating if "filtfilt" mode should be used.}

\item{BatchSize}{An integer indicating the starting batchsize of the trace (chunk size will be optimised for FFT).}

\item{CORES}{An integer indicating what number of cores should be used.}
}
\value{
Returns numeric vector which is the FIR filtered original signal.
}
\description{
This function applies an FIR filter to a signal an returns the filtered trace.
}
