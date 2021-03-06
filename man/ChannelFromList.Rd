% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{ChannelFromList}
\alias{ChannelFromList}
\title{Maximum amplitude channel}
\usage{
ChannelFromList(SpikeCubeList)
}
\arguments{
\item{SpikeCubeList}{A list out of spike shape cubes.}
}
\value{
Returns a list inlcuding the channel number and amplitude.
}
\description{
This function takes a list of single events and computes the channel with max amplitude and amplitude for all units in list.
}
