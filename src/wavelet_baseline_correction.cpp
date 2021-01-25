#include <RcppArmadillo.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' @title Baseline Z-scoring
//' 
//' This function computes the phase lock using the Rho vector length 
//' (strength of the locking) and the circular mean phase from a list containing 
//' wavelet transforms of epochs. The input list has to be flattend (vectorised) 
//' first.
//' 
//' @name BaselineZScore
//' @param InputMat A matrix with time on the x-axis.
//' @param from A starting point for baseline used for z-score.
//' @param to An end point for baseline used for z-score.
//' 
//' @return Returns a matrix with z-scores based on baseline as reference.
//' @export
// [[Rcpp::export]]
arma::mat BaselineZScore(arma::mat InputMat,
                              int from,
                              int to) {
  arma::mat output_mat(InputMat.n_rows, InputMat.n_cols, arma::fill::zeros);
  int end_loop = InputMat.n_rows;
  for(int i=0; i<end_loop; i++) {
    double row_mean, row_sd;
    arma::rowvec tmp_mat = InputMat.submat(i, from-1, i, to-1);
    row_mean = arma::mean(tmp_mat);
    row_sd = arma::stddev(tmp_mat);
    output_mat.row(i) = (InputMat.row(i)-row_mean)/row_sd;
    }
  return output_mat;
}