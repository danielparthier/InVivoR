// [[Rcpp::depends(BH)]]
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG
#include <Rcpp/Benchmark/Timer.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec arma_gaussian(double& sd, double& width, int& SamplingRateOut, arma::vec& SpikeTimings, double& StartTime, double& EndTime) {
  //make gaussian kernel based on mu=0, BinWidth=input, width=input in ms
  arma::vec kernel = arma::normpdf(arma::linspace(-width/2,width/2,SamplingRateOut*width/1000+1), 0, sd);
 // kernel(arma::linspace(0,kernel.index_max(),1));
  kernel /= arma::max(kernel)/(SamplingRateOut/width);
  arma::uvec SpikeIdx = arma::conv_to<arma::uvec>::from(arma::round(SpikeTimings*SamplingRateOut));
  arma::vec tmp_trace = arma::zeros((EndTime-StartTime)*SamplingRateOut);
  tmp_trace.elem(SpikeIdx) += 1;
  return arma::conv(tmp_trace, kernel, "same");
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec arma_gaussian_kernel(double& sd, double& width, int& SamplingRateOut) {
  //make gaussian kernel based on mu=0, sd=input, width=input in ms
  arma::vec kernel = arma::normpdf(arma::linspace(-width/2,width/2,SamplingRateOut*width), 0, sd);
  return kernel;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec arma_gaussian_loop(double& sd, int& SamplingRateOut, arma::vec& SpikeTimings, double& StartTime, double& EndTime) {
  //make gaussian kernel based on mu=0, BinWidth=input
  double width = 15*sd; //make window 15 SD wide on both sides
  //int window = (width*SamplingRateOut+1)/2;
  if(std::abs(std::floor(width*SamplingRateOut/2) - width*SamplingRateOut/2) > 10e-4) {
    width = std::ceil(width*SamplingRateOut/2)*2/SamplingRateOut;
  }
  int kernel_size = std::round(SamplingRateOut*width)+1;
  arma::vec kernel = arma::normpdf(arma::linspace(-width/2,width/2,kernel_size), 0, sd);
  // kernel(arma::linspace(0,kernel.index_max(),1));
  kernel /= arma::sum(kernel);
  int kernel_half_length = kernel.size()/2;
  //kernel.subvec(0,kernel_half_length) = arma::zeros(kernel_half_length);
//  Rcpp::Rcout << arma::zeros<arma::vec>(kernel_half_length) << std::endl;
  arma::vec trace_out = arma::zeros((EndTime-StartTime)*SamplingRateOut);
  arma::vec::iterator i_it = SpikeTimings.begin();
  arma::vec::iterator i_it_end = SpikeTimings.end();
  for(; i_it != i_it_end; ++i_it) {
    if(*i_it > kernel_half_length and *i_it < (trace_out.size()-kernel_half_length)) {
     // Rcpp::Rcout << "in1" << std::endl;
      trace_out.subvec((*i_it-kernel_half_length), (*i_it+kernel_half_length)) += kernel;
     // Rcpp::Rcout << "out1" << std::endl;
    } else if(*i_it <= kernel_half_length) {
      //Rcpp::Rcout << "in2" << std::endl;
      trace_out.subvec(0, (*i_it+kernel_half_length)) += kernel.subvec((kernel_half_length-*i_it),(kernel.size()-1));
      //Rcpp::Rcout << "out2" << std::endl;
    } else if(*i_it >= (trace_out.size()-kernel_half_length)) {
     // Rcpp::Rcout << "in3" << std::endl;
      trace_out.subvec((*i_it-kernel_half_length), trace_out.size()-1) += kernel.subvec(0,(trace_out.size()-*i_it+kernel_half_length-1));
     // Rcpp::Rcout << "out3" << std::endl;
    } else {
      Rcpp::Rcout << "check window size:  " << *i_it << std::endl;
    }
  }
  return trace_out;
}


// [[Rcpp::export]]
arma::vec BAKS(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta){
  omp_set_num_threads(4);
  arma::vec numSum  = arma::zeros(Time.size());
  arma::vec denumSum  = arma::zeros(Time.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    numSum += arma::pow(arma::square(Time-SpikeTimings.at(i))/2+1/beta,-alpha);
    denumSum += arma::pow(arma::square(Time-SpikeTimings.at(i))/2+1/beta,-alpha-0.5);
  }
  arma::vec h  = arma::zeros(Time.size());
  h = (::tgamma(alpha)/::tgamma(alpha+0.5))*(numSum/denumSum);
  arma::vec FiringRate  = arma::zeros(Time.size());
  #pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    FiringRate += 1/(arma::datum::sqrt2pi * h) % arma::exp(-arma::square(Time-SpikeTimings.at(i))/(2 * arma::square(h)));
  }
  return FiringRate;
}


// [[Rcpp::export]]
arma::vec OKS(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta){
  Rcpp::Timer timer;
  timer.step("start");
  omp_set_num_threads(4);
  arma::vec numSum  = arma::zeros(Time.size());
  arma::vec denumSum  = arma::zeros(Time.size());
  timer.step("init");
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    arma::vec tmp = ((Time-SpikeTimings.at(i)) % (Time-SpikeTimings.at(i)))/2 + 1/beta;
    numSum += arma::pow(tmp,-alpha);
    denumSum += arma::pow(tmp,-alpha-0.5);
  }
  timer.step("1stloop");
  arma::vec h  = arma::zeros(Time.size());
  h = (::tgamma(alpha)/::tgamma(alpha+0.5))*(numSum/denumSum);
  arma::vec FiringRate  = arma::zeros(Time.size());
  timer.step("h_init");
  arma::vec h_fast = 1/(arma::datum::sqrt2pi * h);
  h %= h;
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    FiringRate += h_fast % arma::exp(-arma::square(Time-SpikeTimings.at(i))/(2 * h));
  }
  timer.step("2ndloop");
  Rcpp::NumericVector res(timer);   // 
  for (int i=0; i<res.size(); i++) {
    res[i] = res[i];
  }
  return  FiringRate;
}


// [[Rcpp::export]]
arma::vec BAKS_fast(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta){
  omp_set_num_threads(4);
  arma::vec numSum  = arma::zeros(Time.size());
  arma::vec denumSum  = arma::zeros(Time.size());
  arma::vec numSingle = arma::pow(arma::square(arma::linspace(-arma::max(Time), arma::max(Time), Time.size()*2+1))/2+1/beta,-alpha);
  arma::vec denumSingle = arma::pow(arma::square(arma::linspace(-arma::max(Time), arma::max(Time), Time.size()*2+1))/2+1/beta,-alpha-0.5);
  int kernel_half_length = Time.size()+1;
  int trace_length = Time.size();
  //#pragma omp parallel for
  arma::vec::iterator i_it = SpikeTimings.begin();
  arma::vec::iterator i_it_end = SpikeTimings.end();
  for(; i_it != i_it_end; ++i_it) {
    int start_int = kernel_half_length-*i_it*1000;
    int end_int = start_int+trace_length-1;
      numSum += numSingle.subvec(start_int,end_int);
      denumSum += denumSingle.subvec(start_int,end_int);
  }
  arma::vec h  = arma::zeros(trace_length);
  h = (::tgamma(alpha)/::tgamma(alpha+0.5))*(numSum/denumSum);    
  arma::vec FiringRate  = arma::zeros(trace_length);
  arma::vec h_fast = 1/(arma::datum::sqrt2pi * h);
  h %= h;
  #pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    FiringRate += h_fast % arma::exp(-arma::square(Time-SpikeTimings.at(i))/(2 * h));
  }

  return  FiringRate;
}



// [[Rcpp::export]]
arma::vec BAKS_fast_new(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta){
  omp_set_num_threads(4);
  arma::vec numSum  = arma::zeros(Time.size());
  arma::vec denumSum  = arma::zeros(Time.size());
  arma::vec numSingle = arma::pow(arma::square(arma::linspace(-arma::max(Time), arma::max(Time), Time.size()*2+1))/2+1/beta,-alpha);
  arma::vec denumSingle = arma::pow(arma::square(arma::linspace(-arma::max(Time), arma::max(Time), Time.size()*2+1))/2+1/beta,-alpha-0.5);
  int kernel_half_length = Time.size()+1;
  int trace_length = Time.size();
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    int start_int = kernel_half_length-SpikeTimings.at(i)*1000;
    int end_int = start_int+trace_length-1;
    numSum += numSingle.subvec(start_int,end_int);
    denumSum += denumSingle.subvec(start_int,end_int);
  }
  arma::vec h  = arma::zeros(trace_length);
  h = (::tgamma(alpha)/::tgamma(alpha+0.5))*(numSum/denumSum);    
  arma::vec FiringRate  = arma::zeros(trace_length);
  arma::vec h_fast = 1/(arma::datum::sqrt2pi * h);
  h %= h;
#pragma omp parallel for
  for(unsigned int i = 0; i < SpikeTimings.size(); ++i) {
    FiringRate += h_fast % arma::exp(-arma::square(Time-SpikeTimings.at(i))/(2 * h));
  }
  
  return  FiringRate;
}