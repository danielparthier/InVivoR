#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
arma::vec BAKS(arma::vec SpikeTimings,
               double alpha, double& beta,
               double timeStart,
               double timeEnd,
               double SamplingRate = 1e3,
               int CORES = 4) {
  omp_set_num_threads(CORES);
  int OutputLength = (timeEnd-timeStart)*SamplingRate+1;
  arma::vec numSum  = arma::zeros(OutputLength);
  arma::vec denumSum  = arma::zeros(OutputLength);
  arma::vec numSingle = arma::pow(arma::square(arma::linspace(-timeEnd, timeEnd, OutputLength*2+1))/2+1/beta,-alpha);
  arma::vec denumSingle = arma::pow(arma::square(arma::linspace(-timeEnd, timeEnd, OutputLength*2+1))/2+1/beta,-alpha-0.5);
  int kernel_half_length = OutputLength+1;
  arma::vec::iterator i_it = SpikeTimings.begin();
  arma::vec::iterator i_it_end = SpikeTimings.end();
  for(; i_it != i_it_end; ++i_it) {
    int start_int = kernel_half_length-*i_it*1000;
    int end_int = start_int+OutputLength-1;
    numSum += numSingle.subvec(start_int,end_int);
    denumSum += denumSingle.subvec(start_int,end_int);
  }
  arma::vec h  = arma::zeros(OutputLength);
  h = (::tgamma(alpha)/::tgamma(alpha+0.5))*(numSum/denumSum);    
  arma::vec FiringRate  = arma::zeros(OutputLength);
  arma::vec h_fast = 1/(arma::datum::sqrt2pi * h);
  h %= h;
  arma::vec Time = arma::regspace(timeStart, 1/SamplingRate, timeEnd);
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    arma::vec tmp = h_fast % arma::exp(-arma::square(Time-SpikeTimings.at(i))/(2 * h));
#pragma omp critical
    FiringRate += tmp;
  }
  
  return  FiringRate;
}

arma::vec FiringGamma(arma::vec SpikeTimings,
               double alpha,
               double timeStart,
               double timeEnd,
               double SamplingRate = 1e3,
               int CORES = 4) {
  omp_set_num_threads(CORES);
  double preTerm = 1/(alpha*alpha);
  arma::vec FiringRate  = arma::zeros((timeEnd-timeStart)*SamplingRate+1);
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    arma::vec xTmp = arma::regspace(0, 1/SamplingRate, (timeEnd-SpikeTimings.at(i)));
    arma::vec tmp = preTerm*xTmp%arma::exp(-xTmp/alpha);
#pragma omp critical
    FiringRate.tail(xTmp.size()) += tmp;
  }
  return FiringRate;
}

arma::vec FiringGaussian(arma::vec SpikeTimings,
                      double sigma,
                      double timeStart,
                      double timeEnd,
                      double SamplingRate = 1e3,
                      int CORES = 4) {
  omp_set_num_threads(CORES);
  arma::vec FiringRate  = arma::zeros((timeEnd-timeStart)*SamplingRate+1);
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    arma::vec tmp = arma::normpdf(arma::linspace(-SpikeTimings.at(i),timeEnd-SpikeTimings.at(i), FiringRate.size()), 0, sigma);
#pragma omp critical   
    FiringRate += tmp;
  }
  return FiringRate;
}



//' ERP (event-related potential) extraction
//' 
//' This function returns a matrix with extracted traces for any given range. 
//' The range is has to be provided in form of onset indeces and end indeces. 
//' Additional information can be provided to adjust for sampling differences 
//' between range points and trace sampling frequency. If required the window 
//' is proportionally elongated to include Pre/Post times.
//'
//' @param SpikeTimes A numeric vector with spike times in seconds.
//' @param timeStart A double for the start time for which the rate should be computed.
//' @param timeEnd A double for the end time for which the rate should be computed.
//' @param sigma A double indicating sigma of the gausian kernel in seconds.
//' @param alpha A double indicating alpha of the gamma kernel in seconds.
//' @param useBAKS A indicating whether Bayesian Adaptive Kernel Smoother should be used.
//' @param BAKSalpha A double indicating the prior for alpha in BAKS (default = 4).
//' @param BAKSbeta A double indicating the prior for beta in BAKS (default = 4).
//' @param SamplingRate A double indicating the sampling rate (default = 1000).
//' @param CORES An integer indicating how many core should be used (default = 4).
//' @return Returns a vector with firing rate.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector FiringRate(arma::vec& SpikeTimes,
                               Rcpp::Nullable<double> timeStart = R_NilValue,
                               Rcpp::Nullable<double> timeEnd = R_NilValue,
                               Rcpp::Nullable<double> sigma = R_NilValue,
                               Rcpp::Nullable<double> alpha = R_NilValue,
                               bool useBAKS = false,
                               double BAKSalpha = 4,
                               double BAKSbeta = 4,
                               double SamplingRate = 1e3,
                               int CORES = 4L) {
  double timeEndtmp, timeStarttmp;
  if(timeStart.isNotNull() && timeEnd.isNotNull()) {
    timeEndtmp = Rcpp::as<double>(timeEnd);
    timeStarttmp = Rcpp::as<double>(timeStart);
  } else {
    timeEndtmp = arma::max(SpikeTimes);
    timeStarttmp = arma::min(SpikeTimes);
  }
  arma::vec FreqOut;
  if(sigma.isNotNull() || (sigma.isNull() && alpha.isNull()) & (useBAKS == false)) {
    double sd;
    if(sigma.isNull() && alpha.isNull()) {
      sd = 0.1;
    } else {
      sd = Rcpp::as<double>(sigma);
    }
    FreqOut = FiringGaussian(SpikeTimes, sd, timeStarttmp, timeEndtmp,SamplingRate, CORES);
  } else if(alpha.isNotNull() && (useBAKS == false)) {
    double alphatmp = Rcpp::as<double>(alpha);
    FreqOut = FiringGamma(SpikeTimes, alphatmp, timeStarttmp, timeEndtmp,SamplingRate, CORES);
  } else if(useBAKS) {
    FreqOut = BAKS(SpikeTimes, BAKSalpha, BAKSbeta, timeStarttmp, timeEndtmp, SamplingRate, CORES);
  }
  return Rcpp::NumericVector(FreqOut.begin(),FreqOut.end());
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::NumericVector FiringRateSparse(arma::vec& SpikeTimes,
                               Rcpp::Nullable<double> sigma = R_NilValue,
                               Rcpp::Nullable<double> alpha = R_NilValue,
                               double SamplingRate = 1e3) {
  arma::vec FreqOut = arma::zeros(SpikeTimes.size());
  if(sigma.isNotNull() || (sigma.isNull() && alpha.isNull())) {
    double sd;
    if(sigma.isNull() && alpha.isNull()) {
      sd = 0.1;
    } else {
      sd = Rcpp::as<double>(sigma);
    }
    for(unsigned int i = 0; i < SpikeTimes.size(); ++i) {
      FreqOut += arma::normpdf(SpikeTimes-SpikeTimes.at(i), 0, sd);
    }
  }
  if(alpha.isNotNull()) {
    double alphatmp = Rcpp::as<double>(alpha);
    double preTerm = 1/(alphatmp*alphatmp);
    arma::vec tmp = arma::zeros(size(SpikeTimes));
    for(unsigned int i = 0; i < SpikeTimes.size(); ++i) {
      tmp = preTerm*(SpikeTimes-SpikeTimes.at(i))%arma::exp(-(SpikeTimes-SpikeTimes.at(i))/alphatmp);
      tmp.elem( arma::find(tmp < 0) ).zeros();
      FreqOut += tmp;
      }
  }
  return Rcpp::NumericVector(FreqOut.begin(),FreqOut.end());
}