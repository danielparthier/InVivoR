#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' Phase lock analysis
//' 
//' This function computes the phase lock using the Rho vector length 
//' (strength of the locking) and the circular mean phase from a cube containing 
//' wavelet transforms of epochs.
//' 
//' @param x A cube of phases (radians) with slices as different ERPs.
//' @param CORES An int indicating the number of threads used (default = 1).
//' @return Returns a list containing a matrix with Rho vector lengths and a matrix with the corresponding circular mean.
//' @export
// [[Rcpp::export]]
Rcpp::List PhaseListAnalysis(const arma::cube& x,
                             const int& CORES = 1) {
  omp_set_num_threads(CORES);
  int DIM_X = x.n_rows;
  int DIM_Y = x.n_cols;
  arma::mat OutputRho = arma::mat(DIM_X, DIM_Y, arma::fill::zeros);
  arma::mat OutputMean = arma::mat(DIM_X, DIM_Y, arma::fill::zeros);
  #pragma omp parallel for shared(x, OutputRho, OutputMean) schedule(static) 
  for(int i = 0; i < DIM_X; ++i) {
    for(int j = 0; j < DIM_Y; ++j) {
      arma::rowvec tmp_vec = x.tube(i, j);
      double tmpX = arma::mean(arma::sin(tmp_vec));
      double tmpY = arma::mean(arma::cos(tmp_vec));
      // changed from OutputRho(i, j) = sqrt((pow(tmpX, 2)+pow(tmpY, 2)));
      OutputRho(i, j) = sqrt(tmpX*tmpX+tmpY*tmpY);
      OutputMean(i, j) = atan2(tmpX, tmpY);
    }  
  }
  return Rcpp::List::create(Rcpp::Named("Rho") = OutputRho,
                            Rcpp::Named("Mean") = OutputMean);
}

// [[Rcpp::depends(RcppArmadillo)]]
//' Matrix shuffle
//' 
//' This function shuffles the data independently of dimensions without 
//' resampling from a input matrix. It returns the probability for a any 
//' given value in a matrix to be larger than a random sample.
//' 
//' @param x A matrix.
//' @param SHUFFLES An int indicating the number of shuffles.
//' @param CORES An int indicating the number of threads used (default = 1).
//' @return Returns a matrix indicating the probability of value being larger than shuffled data.
//' @export
// [[Rcpp::export]]
arma::mat PhaseListAnalysisShuffle(arma::mat& x,
                                   const int SHUFFLES = 200,
                                   int CORES = 1) {
  omp_set_num_threads(CORES);
  int DIM_X = x.n_rows;
  int DIM_Y = x.n_cols;
  arma::mat OutputSig = arma::mat(DIM_X, DIM_Y, arma::fill::zeros);
  #pragma omp parallel for shared(x, OutputSig) schedule(static) 
  for(int i = 0; i < SHUFFLES; ++i) {
    OutputSig = OutputSig+(x>arma::shuffle(x, 1));
  }
  return OutputSig/SHUFFLES;
}

// [[Rcpp::depends(RcppArmadillo)]]
//' Matrix resample
//' 
//' This function shuffles the data independently of dimensions with 
//' resampling from a input matrix. It returns the probability for a any 
//' given value in a matrix to be larger than a random sample.
//' 
//' @param x A matrix.
//' @param SHUFFLES An int indicating the number of shuffles.
//' @param CORES An int indicating the number of threads used (default = 1).
//' @return Returns a matrix indicating the probability of value being larger than shuffled data.
//' @export
// [[Rcpp::export]]
arma::mat PhaseListAnalysisResample(arma::mat& x,
                                    const int SHUFFLES = 200,
                                    const int CORES = 1) {
  omp_set_num_threads(CORES);
  int DIM_X = x.n_rows;
  int DIM_Y = x.n_cols;
  arma::mat OutputSig = arma::mat(DIM_X, DIM_Y, arma::fill::zeros);
  arma::mat ShuffleMat = arma::mat(DIM_X, DIM_Y, arma::fill::zeros);
  #pragma omp parallel for shared(x, OutputSig, ShuffleMat, DIM_X, DIM_Y) schedule(static) 
  for(int shuffle_run = 0; shuffle_run < SHUFFLES; ++shuffle_run) {
    arma::vec VecRnd = arma::randi<arma::vec>(DIM_X*DIM_Y, arma::distr_param(0, DIM_Y*DIM_X));
    int k = 0;
    for(int i = 0;i < DIM_X; ++i){
      for(int j = 0;j < DIM_Y; ++j){
        int VecIdx = VecRnd.at(k);
        ShuffleMat.at(i,j) = x.at(VecIdx);
        ++k;
      }
    }
    OutputSig = OutputSig+(x>ShuffleMat);
  }
  return OutputSig/SHUFFLES;
}