#' Confidence interval poisson
#' 
#' This function computes a confidence interval for a series of counts.
#' The data is convolved with a inverted gaussian to calculate a weighted mean as lamda.
#' From the new lamda the confidence interval is computed using the chisq distribution as approximation.
#' @param CountVector An integer vector of counts
#' @param conf.level A double indication the confidence interval.
#' @param SD Parameter of inverted gaussian.
#' @param CentreMin Parameter of inverted gaussian.
#' @param KernelSize Parameter of inverted gaussian.
#'
#' @return
#' @export
#'
#' @examples
ConfIntPoissonR <- function(CountVector,
                     conf.level = 0.95,
                     SD = 0.6,
                     CentreMin = 0.6,
                     KernelSize = 65) {
  GaussKernel <- stats::dnorm(x = -KernelSize:KernelSize, mean = 0, sd = SD)
  GaussKernel <- GaussKernel/max(GaussKernel)
  GaussKernel <- 1-(GaussKernel*(1-CentreMin))
  GaussKernel <- GaussKernel/mean(GaussKernel)
  CountVector <- c(rep_len(CountVector[1], KernelSize), CountVector, rep_len(CountVector[length(CountVector)], KernelSize))
  LamdaVec <- stats::convolve(x = CountVector, y = GaussKernel, type = "filter")/(2*KernelSize+1)
  alpha <- 1-conf.level
  return(cbind((0.5 * stats::qchisq(alpha/2, 2*LamdaVec)), (0.5 * stats::qchisq(1-alpha/2, 2*LamdaVec+2))))
}