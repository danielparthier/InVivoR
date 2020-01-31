#include <RcppArmadillo.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' Phase lock analysis
//' 
//' This function computes the phase lock using the Rho vector length 
//' (strength of the locking) and the circular mean phase from a list containing 
//' wavelet transforms of epochs. The input list has to be flattend (vectorised) 
//' first.
//' 
//' @param input_mat A matrix with time on the x-axis.
//' @param from_point A starting point for baseline used for z-score.
//' @param to_point An end point for baseline used for z-score.
//' @return Returns a matrix with z-scores based on baseline as reference.
//' @export
// [[Rcpp::export]]
arma::mat mat_baseline_zscore(arma::mat input_mat,
                              int from_point,
                              int to_point) {
  arma::mat output_mat(input_mat.n_rows, input_mat.n_cols, arma::fill::zeros);
  int end_loop = input_mat.n_rows;
  for(int i=0; i<end_loop; i++) {
    double row_mean, row_sd;
    arma::rowvec tmp_mat = input_mat.submat(i, from_point-1, i, to_point-1);
    row_mean = arma::mean(tmp_mat);
    row_sd = arma::stddev(tmp_mat);
    output_mat.row(i) = (input_mat.row(i)-row_mean)/row_sd;
    }
  return output_mat;
}