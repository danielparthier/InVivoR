#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

//' Spike cut out
//' 
//' This function extracts spike shape matrices from an ampliflier matrix (rows = channels, columns = time).
//'
//' @param AmpMatrix A matrix with signal from the amplifier (rows = channels, columns = time).
//' @param SpikeIdx A matrix with signal from the amplifier (rows = channels, columns = time).
//' @param WINDOW An integer indicating the edges (default = 20).
//' @return Returns a vector with position of maximum amplitude and amplitude itself.
//' @export
// [[Rcpp::export]]
arma::cube SpikeCut(const arma::mat& AmpMatrix,
                    const arma::vec& SpikeIdx,
                    const int& WINDOW = 20){
  arma::vec SpikeIdxUse = SpikeIdx(arma::find(SpikeIdx>WINDOW and SpikeIdx<(AmpMatrix.n_cols-WINDOW)));
  arma::cube outCube(AmpMatrix.n_rows, WINDOW*2+1, SpikeIdxUse.size(), arma::fill::zeros);
  for(unsigned int i = 0; i < SpikeIdxUse.n_elem; ++i){
    outCube.slice(i) = AmpMatrix.cols(SpikeIdxUse.at(i)-WINDOW, SpikeIdxUse.at(i)+WINDOW);
  }
  return outCube;
}

//' Spike median matrix
//' 
//' This function computes the median shape of spikes from a single spike cube.
//'
//' @param SpikeCube A cube containing matrices with isolated spikes (rows = channels, columns = time).
//' @return Returns a matrix with median spike shapes (rows = channels, columns = time).
//' @export
// [[Rcpp::export]]
arma::mat SpikeMed(const arma::cube& SpikeCube){
  arma::mat MedMat = arma::mat(SpikeCube.n_rows, SpikeCube.n_cols, arma::fill::zeros);
  for(unsigned int i = 0; i < SpikeCube.n_rows; ++i){
    for(unsigned int j = 0; j < SpikeCube.n_cols; ++j){
      arma::vec tmp = SpikeCube.tube(i,j);
      MedMat(i,j) = arma::median(tmp); 
    }
  }
  // correct drift by subtracting median in channel
  for(unsigned int i = 0; i < SpikeCube.n_rows; ++i){
    MedMat.row(i) -= arma::median(MedMat.row(i));
  }
  return MedMat;
}


//' Maximum amplitude channel
//' 
//' This function uses the median spike shape to find channel with maximum amplitude.
//' As input the function requires the previously computed median spike matrix with dimensions of n.row = clusters and n.col = channel count
//'
//' @param MedianSpikeMat A median matrix from arma_apply_median.
//' @return Returns a vector with position of maximum amplitude and amplitude itself.
//' @export
// [[Rcpp::export]]
arma::rowvec MaxChannel(const arma::mat& MedianSpikeMat) {
  int channel_nr = MedianSpikeMat.n_rows;
  arma::rowvec channel_out(2, arma::fill::zeros);
  arma::vec min_peak_amp(channel_nr, arma::fill::zeros);
  for(int i=0; i<channel_nr; i++) {
    min_peak_amp(i) = arma::min(MedianSpikeMat.row(i));
  }
  channel_out(0) = arma::index_min(min_peak_amp)+1;
  channel_out(1) = arma::min(min_peak_amp);
  return channel_out;
}

//' Maximum amplitude channel
//' 
//' This function takes a list of single events and computes the channel with max amplitude and amplitude for all units in list.
//'
//' @param SpikeCubeList A list out of spike shape cubes.
//' @return Returns a list inlcuding the channel number and amplitude.
//' @export
// [[Rcpp::export]]
Rcpp::List ChannelFromList(const Rcpp::List& SpikeCubeList) {
  int end_i = SpikeCubeList.size();
  arma::mat out_mat = arma::mat(end_i, 2, arma::fill::zeros);
  for(int i=0; i<end_i; i++) {
    out_mat.row(i) = MaxChannel(SpikeMed(SpikeCubeList[i]));
  }
  return Rcpp::List::create(Rcpp::Named("ChannelNr") = out_mat.col(0),
                            Rcpp::Named("Amplitude") = out_mat.col(1));
}

//' Spike location channel
//' 
//' This function is a wrapper to extract the channel location for any given unit based on the spike timings. It uses the spike 
//' indices, unit numbers, and the amplifier matrix (row = channels, Columns = time). It avoids using an apply function and 
//' directly accesses the data matrix.
//'
//' @param SpikeIdx A numeric vector with spike indices.
//' @param Units A numeric vector unit numbers.
//' @param AmpMatrix A matrix with signal from the amplifier (rows = channels, columns = time).
//' @return Returns a list inlcuding the channel number, amplitude and unit number.
//' @export
// [[Rcpp::export]]
Rcpp::List UnitChannel(const arma::vec& SpikeIdx,
                       const arma::vec& Units,
                       const arma::mat& AmpMatrix,
                       const int& WINDOW = 20) {
  int ChannelCount = AmpMatrix.n_rows;
  arma::vec UnitsUnique = arma::unique(Units);
  arma::mat out_mat = arma::mat(UnitsUnique.n_elem, 3, arma::fill::zeros);
  arma::cube SpikeShape = arma::cube(AmpMatrix.n_rows, 2*WINDOW+1, UnitsUnique.n_elem, arma::fill::zeros);
  for(long int i=0; i<UnitsUnique.n_elem; i++) {
    arma::vec SpikeVec = SpikeIdx.elem(find(Units == UnitsUnique.at(i)));
    arma::uvec SpikeSelectIdx = arma::conv_to<arma::uvec>::from(SpikeVec);
    for(long int j=0; j<AmpMatrix.n_rows; ++j){
      for(long int k = -WINDOW; k < WINDOW+1; ++k){
        arma::uvec tmpIDX = (SpikeSelectIdx+k)*ChannelCount+j;
        SpikeShape.at(j,k+WINDOW,i) = arma::median(AmpMatrix.elem(tmpIDX));
      }
      SpikeShape.slice(i).row(j) -= arma::median(SpikeShape.slice(i).row(j));
    } 
    out_mat.submat(i, 0, i, 1) = MaxChannel(SpikeShape.slice(i));
    out_mat.at(i, 2) = UnitsUnique.at(i);
  }
  return Rcpp::List::create(Rcpp::Named("ChannelNr") = out_mat.col(0),
                            Rcpp::Named("Amplitude") = out_mat.col(1),
                            Rcpp::Named("UnitNr") = out_mat.col(2),
                            Rcpp::Named("SpikeShape") = SpikeShape);
}
