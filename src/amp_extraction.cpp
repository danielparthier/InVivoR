#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Maximum Amplitude Channel
//' 
//' This function uses the median spike shape to find channel with maximum amplitude.
//' As input the function requires the previously computed median spike matrix with dimensions of n.row = clusters and n.col = channel count
//'
//' @param median_input_mat A median matrix from arma_apply_median.
//' @return Returns a vector with position of maximum amplitude and amplitude itself.
//' @export
// [[Rcpp::export]]
arma::rowvec max_channel(arma::mat median_input_mat) {
  int channel_nr = median_input_mat.n_cols;
  arma::rowvec channel_out(2, arma::fill::zeros);
  arma::vec min_peak_amp(channel_nr, arma::fill::zeros);
  for(int i=0; i<channel_nr; i++) {
    min_peak_amp(i) = arma::min(median_input_mat.col(i));
  }
  channel_out(0) = arma::index_min(min_peak_amp)+1;
  channel_out(1) = arma::min(min_peak_amp);
  return channel_out;
}

//' Median Spike
//' 
//' This function uses an amplitude matrix from single spikes in channel.
//'
//' @param input_mat An input matrix with amplitudes for single spikes.
//' @return Returns a vector with length of concatenated samples.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector arma_apply_median(arma::mat input_mat) {
  arma::vec a(input_mat.n_rows, arma::fill::zeros);
  int end_loop = input_mat.n_rows;
  for(int i=0; i<end_loop; i++) {
    a(i) = arma::median(input_mat.row(i));
  }
  return Rcpp::NumericVector(a.begin(),a.end());
}

//' Spike extraction loop
//' 
//' This function uses an amplitude matrix from single spikes in channel to output baseline corrected median spikes.
//'
//' @param input_mat_raw A numeric vector from a matrix with amplitudes for single spikes.
//' @param channel_nr An int indicating the number of recorded channels.
//' @return Returns a numeric spike vector with corrected baseline for single channels.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector arma_spike_extraction_loop(arma::vec input_mat_raw, int channel_nr) {
  //Rcpp::Rcout << channel_nr << "  arma_spike_extraction_loop "<< std::endl;
  int row_numbers = input_mat_raw.n_elem/channel_nr;
  arma::mat input_mat = arma::mat(input_mat_raw.memptr(), channel_nr, row_numbers, false, false);
  input_mat = arma::trans(input_mat);
  arma::vec median_vec(input_mat.n_rows, arma::fill::zeros);
  arma::uvec first_idx = arma::conv_to < arma::uvec >::from(arma::regspace(0,10));
  int end_loop = input_mat.n_rows;
  for(int i=0; i<end_loop; i++) {
    median_vec(i) = arma::median(input_mat.row(i));
  }
  arma::mat tmp_mat(input_mat.n_rows, input_mat.n_cols, arma::fill::zeros);
  end_loop = input_mat.n_cols;
  for(int k=0; k<end_loop; k++) {
    arma::vec tmp_vec = input_mat.col(k)-median_vec;
    tmp_mat.col(k) = tmp_vec-arma::median(tmp_vec(first_idx));
  }
  return Rcpp::NumericVector(tmp_mat.begin(),tmp_mat.end());
}

//' Maximum amplitude channel
//' 
//' This function takes a list of single events and computes the channel with max amplitude and amplitude for all units in list.
//'
//' @param spike_shape_list A numeric vector from a matrix with amplitudes for single spikes.
//' @param channel_nr An int indicating the number of recorded channels.
//' @return Returns a numeric matrix inlcuding the channel number and amplitude.
//' @export
// [[Rcpp::export]]
arma::mat chan_out(Rcpp::List spike_shape_list, int channel_nr) {
  int end_i = spike_shape_list.size();
  arma::mat out_mat = arma::mat(end_i, 2, arma::fill::zeros);
  for(int i=0; i<end_i; i++) {
    arma::vec tmp_vec = arma_apply_median(spike_shape_list[i]);
    arma::mat input_mat = arma::mat(tmp_vec.memptr(), tmp_vec.n_elem/channel_nr, channel_nr, false, false);
    out_mat.row(i) = max_channel(input_mat);
  }
  return out_mat;
}