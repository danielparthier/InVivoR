// Function to compute Modulation index from Power Matrix and Phase Matrix
// dimensions: rows = time, columns = periods
#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

//' @useDynLib InVivoR
//' @importFrom Rcpp sourceCpp

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

//' Modulation index analysis
//' 
//' This function returns a list including a modulation index matrix, the phase frequency and the power frequency.
//' there are leading NA, marking the leadings NA as TRUE and
//' everything else as FALSE.
//'
//' @param PowerMatRaw A matrix with power.
//' @param PhaseMatRaw A matrix with phase.
//' @param PowerPeriods A double vector with unique periods (1/Frequency) for power matrix.
//' @param PhasePeriods A double vector with unique periods (1/Frequency) for phase matrix.
//' @param BIN_NUMBER An integer indicating in how many parts phase should be split.
//' @param PHASE_FREQ_START A double indicating starting frequency (1/period) for phase.
//' @param PHASE_FREQ_END A double indicating end frequency (1/period) for phase.
//' @param POWER_FREQ_START A double indicating starting frequency (1/period) for power.
//' @param POWER_FREQ_END A A double indicating end frequency (1/period) for power.
//' @param CORES An integer indicating number of threads used (multicore support).
//' @return Returns a list containing the modulation index matrix and vectors containing its axis (Powerfrequency and Phasefrequency)
//' @export
// [[Rcpp::export]]
Rcpp::List MI(arma::mat& PowerMatRaw,
              arma::mat& PhaseMatRaw,
              arma::vec& PowerPeriods,
              arma::vec& PhasePeriods,
              const double& BIN_NUMBER,
              const double& PHASE_FREQ_START,
              const double& PHASE_FREQ_END,
              const double& POWER_FREQ_START,
              const double& POWER_FREQ_END,
              const int& CORES) {
  omp_set_num_threads(CORES);
  const int PHASE_START = arma::as_scalar(arma::find((arma::round(1/PhasePeriods*1e8)/1e8)>=PHASE_FREQ_START, 1, "first"));
  const int PHASE_END = arma::as_scalar(arma::find((arma::round(1/PhasePeriods*1e8)/1e8)<=PHASE_FREQ_END, 1, "last"));
  const int POWER_START = arma::as_scalar(arma::find((arma::round(1/PowerPeriods*1e8)/1e8)>=POWER_FREQ_START, 1, "first"));
  const int POWER_END = arma::as_scalar(arma::find((arma::round(1/PowerPeriods*1e8)/1e8)<=POWER_FREQ_END, 1, "last"));
  const int PHASE_DIM = PHASE_END-PHASE_START+1;
  const int POWER_DIM = POWER_END-POWER_START+1;
  arma::vec PhaseFreq = PhasePeriods.subvec(PHASE_START,PHASE_END);
  arma::vec PowerFreq = PowerPeriods.subvec(POWER_START,POWER_END);
  const int END_I = PhaseMatRaw.n_rows;
  arma::mat PowerMat = arma::mat(PowerMatRaw.memptr(), PowerMatRaw.n_rows, PowerMatRaw.n_cols, true, false).t();
  arma::mat PhaseMat = arma::mat(PhaseMatRaw.memptr(), PhaseMatRaw.n_rows, PhaseMatRaw.n_cols, true, false).t();
  const double LOG_BIN = log(BIN_NUMBER);
  arma::vec MeanPower;
  arma::mat MIMat = arma::mat(POWER_DIM, PHASE_DIM, arma::fill::zeros);
  arma::vec PiSeq = arma::vec(END_I, arma::fill::zeros);
  MeanPower = arma::zeros<arma::vec>(BIN_NUMBER);
  #pragma omp parallel for shared(MIMat, PowerMat, PhaseMat, BIN_NUMBER) private(MeanPower, PiSeq) schedule(static) default(none)
  for(int i=PHASE_START; i<PHASE_END+1; ++i) {
    MeanPower = arma::zeros<arma::vec>(BIN_NUMBER);
    PiSeq = arma::round(PhaseMat.unsafe_col(i)/arma::datum::pi*BIN_NUMBER/2)+BIN_NUMBER/2;
    for(int j=POWER_START; j<POWER_END+1; ++j) {
      arma::vec MeanPowerCol = PowerMat.unsafe_col(j);
      for(int k=0; k<BIN_NUMBER; ++k) {
        arma::uvec PiFind = arma::find(PiSeq==k);
        if(PiFind.n_elem>0) {
          MeanPower(k) = arma::mean(MeanPowerCol.elem(PiFind));
        }
      }
      const double MeanPowerSum = arma::sum(MeanPower);
      MeanPower /= MeanPowerSum;
      MIMat.at(j-POWER_START,i-PHASE_START) = arma::sum(MeanPower%arma::log(MeanPower*BIN_NUMBER))/LOG_BIN;
    }
  }
 return Rcpp::List::create(Rcpp::Named("MI") = MIMat,
                           Rcpp::Named("PhaseFrequency") = 1/Rcpp::NumericVector(PhaseFreq.begin(),PhaseFreq.end()),
                           Rcpp::Named("PowerFrequency") = 1/Rcpp::NumericVector(PowerFreq.begin(),PowerFreq.end()));
}