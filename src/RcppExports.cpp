// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// BinaryFileAccess
arma::cube BinaryFileAccess(const std::string& FILENAME, arma::vec& spikePoints, const int& WINDOW, const unsigned int& CHANNELCOUNT, const unsigned int& CACHESIZE, const unsigned int& BYTECODE);
RcppExport SEXP _InVivoR_BinaryFileAccess(SEXP FILENAMESEXP, SEXP spikePointsSEXP, SEXP WINDOWSEXP, SEXP CHANNELCOUNTSEXP, SEXP CACHESIZESEXP, SEXP BYTECODESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type FILENAME(FILENAMESEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type spikePoints(spikePointsSEXP);
    Rcpp::traits::input_parameter< const int& >::type WINDOW(WINDOWSEXP);
    Rcpp::traits::input_parameter< const unsigned int& >::type CHANNELCOUNT(CHANNELCOUNTSEXP);
    Rcpp::traits::input_parameter< const unsigned int& >::type CACHESIZE(CACHESIZESEXP);
    Rcpp::traits::input_parameter< const unsigned int& >::type BYTECODE(BYTECODESEXP);
    rcpp_result_gen = Rcpp::wrap(BinaryFileAccess(FILENAME, spikePoints, WINDOW, CHANNELCOUNT, CACHESIZE, BYTECODE));
    return rcpp_result_gen;
END_RCPP
}
// decimate
Rcpp::NumericVector decimate(arma::vec& SIGNAL, const arma::vec& FIR_FILTER, const int& M);
RcppExport SEXP _InVivoR_decimate(SEXP SIGNALSEXP, SEXP FIR_FILTERSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SIGNAL(SIGNALSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type FIR_FILTER(FIR_FILTERSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(decimate(SIGNAL, FIR_FILTER, M));
    return rcpp_result_gen;
END_RCPP
}
// arma_gaussian
arma::vec arma_gaussian(double& sd, double& width, int& SamplingRateOut, arma::vec& SpikeTimings, double& StartTime, double& EndTime);
RcppExport SEXP _InVivoR_arma_gaussian(SEXP sdSEXP, SEXP widthSEXP, SEXP SamplingRateOutSEXP, SEXP SpikeTimingsSEXP, SEXP StartTimeSEXP, SEXP EndTimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double& >::type sd(sdSEXP);
    Rcpp::traits::input_parameter< double& >::type width(widthSEXP);
    Rcpp::traits::input_parameter< int& >::type SamplingRateOut(SamplingRateOutSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimings(SpikeTimingsSEXP);
    Rcpp::traits::input_parameter< double& >::type StartTime(StartTimeSEXP);
    Rcpp::traits::input_parameter< double& >::type EndTime(EndTimeSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_gaussian(sd, width, SamplingRateOut, SpikeTimings, StartTime, EndTime));
    return rcpp_result_gen;
END_RCPP
}
// arma_gaussian_kernel
arma::vec arma_gaussian_kernel(double& sd, double& width, int& SamplingRateOut);
RcppExport SEXP _InVivoR_arma_gaussian_kernel(SEXP sdSEXP, SEXP widthSEXP, SEXP SamplingRateOutSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double& >::type sd(sdSEXP);
    Rcpp::traits::input_parameter< double& >::type width(widthSEXP);
    Rcpp::traits::input_parameter< int& >::type SamplingRateOut(SamplingRateOutSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_gaussian_kernel(sd, width, SamplingRateOut));
    return rcpp_result_gen;
END_RCPP
}
// arma_gaussian_loop
arma::vec arma_gaussian_loop(double& sd, int& SamplingRateOut, arma::vec& SpikeTimings, double& StartTime, double& EndTime);
RcppExport SEXP _InVivoR_arma_gaussian_loop(SEXP sdSEXP, SEXP SamplingRateOutSEXP, SEXP SpikeTimingsSEXP, SEXP StartTimeSEXP, SEXP EndTimeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double& >::type sd(sdSEXP);
    Rcpp::traits::input_parameter< int& >::type SamplingRateOut(SamplingRateOutSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimings(SpikeTimingsSEXP);
    Rcpp::traits::input_parameter< double& >::type StartTime(StartTimeSEXP);
    Rcpp::traits::input_parameter< double& >::type EndTime(EndTimeSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_gaussian_loop(sd, SamplingRateOut, SpikeTimings, StartTime, EndTime));
    return rcpp_result_gen;
END_RCPP
}
// BAKS
arma::vec BAKS(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta);
RcppExport SEXP _InVivoR_BAKS(SEXP SpikeTimingsSEXP, SEXP TimeSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimings(SpikeTimingsSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Time(TimeSEXP);
    Rcpp::traits::input_parameter< double& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(BAKS(SpikeTimings, Time, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// OKS
arma::vec OKS(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta);
RcppExport SEXP _InVivoR_OKS(SEXP SpikeTimingsSEXP, SEXP TimeSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimings(SpikeTimingsSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Time(TimeSEXP);
    Rcpp::traits::input_parameter< double& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(OKS(SpikeTimings, Time, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// BAKS_fast
arma::vec BAKS_fast(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta);
RcppExport SEXP _InVivoR_BAKS_fast(SEXP SpikeTimingsSEXP, SEXP TimeSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimings(SpikeTimingsSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Time(TimeSEXP);
    Rcpp::traits::input_parameter< double& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(BAKS_fast(SpikeTimings, Time, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// BAKS_fast_new
arma::vec BAKS_fast_new(arma::vec& SpikeTimings, arma::vec& Time, double& alpha, double& beta);
RcppExport SEXP _InVivoR_BAKS_fast_new(SEXP SpikeTimingsSEXP, SEXP TimeSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimings(SpikeTimingsSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type Time(TimeSEXP);
    Rcpp::traits::input_parameter< double& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(BAKS_fast_new(SpikeTimings, Time, alpha, beta));
    return rcpp_result_gen;
END_RCPP
}
// MI
Rcpp::List MI(arma::mat& PowerMatRaw, arma::mat& PhaseMatRaw, arma::vec& PowerPeriods, arma::vec& PhasePeriods, const double& BIN_NUMBER, const double& PHASE_FREQ_START, const double& PHASE_FREQ_END, const double& POWER_FREQ_START, const double& POWER_FREQ_END, const int& CORES);
RcppExport SEXP _InVivoR_MI(SEXP PowerMatRawSEXP, SEXP PhaseMatRawSEXP, SEXP PowerPeriodsSEXP, SEXP PhasePeriodsSEXP, SEXP BIN_NUMBERSEXP, SEXP PHASE_FREQ_STARTSEXP, SEXP PHASE_FREQ_ENDSEXP, SEXP POWER_FREQ_STARTSEXP, SEXP POWER_FREQ_ENDSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type PowerMatRaw(PowerMatRawSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type PhaseMatRaw(PhaseMatRawSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type PowerPeriods(PowerPeriodsSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type PhasePeriods(PhasePeriodsSEXP);
    Rcpp::traits::input_parameter< const double& >::type BIN_NUMBER(BIN_NUMBERSEXP);
    Rcpp::traits::input_parameter< const double& >::type PHASE_FREQ_START(PHASE_FREQ_STARTSEXP);
    Rcpp::traits::input_parameter< const double& >::type PHASE_FREQ_END(PHASE_FREQ_ENDSEXP);
    Rcpp::traits::input_parameter< const double& >::type POWER_FREQ_START(POWER_FREQ_STARTSEXP);
    Rcpp::traits::input_parameter< const double& >::type POWER_FREQ_END(POWER_FREQ_ENDSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(MI(PowerMatRaw, PhaseMatRaw, PowerPeriods, PhasePeriods, BIN_NUMBER, PHASE_FREQ_START, PHASE_FREQ_END, POWER_FREQ_START, POWER_FREQ_END, CORES));
    return rcpp_result_gen;
END_RCPP
}
// PhaseListAnalysis
Rcpp::List PhaseListAnalysis(arma::vec& x, int& DIM_X, int& DIM_Y, int& DIM_Z, const int& CORES);
RcppExport SEXP _InVivoR_PhaseListAnalysis(SEXP xSEXP, SEXP DIM_XSEXP, SEXP DIM_YSEXP, SEXP DIM_ZSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int& >::type DIM_X(DIM_XSEXP);
    Rcpp::traits::input_parameter< int& >::type DIM_Y(DIM_YSEXP);
    Rcpp::traits::input_parameter< int& >::type DIM_Z(DIM_ZSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(PhaseListAnalysis(x, DIM_X, DIM_Y, DIM_Z, CORES));
    return rcpp_result_gen;
END_RCPP
}
// PhaseListAnalysisShuffle
arma::mat PhaseListAnalysisShuffle(arma::mat& x, const int& DIM_X, const int& DIM_Y, const int& SHUFFLES, int& CORES);
RcppExport SEXP _InVivoR_PhaseListAnalysisShuffle(SEXP xSEXP, SEXP DIM_XSEXP, SEXP DIM_YSEXP, SEXP SHUFFLESSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int& >::type DIM_X(DIM_XSEXP);
    Rcpp::traits::input_parameter< const int& >::type DIM_Y(DIM_YSEXP);
    Rcpp::traits::input_parameter< const int& >::type SHUFFLES(SHUFFLESSEXP);
    Rcpp::traits::input_parameter< int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(PhaseListAnalysisShuffle(x, DIM_X, DIM_Y, SHUFFLES, CORES));
    return rcpp_result_gen;
END_RCPP
}
// PhaseListAnalysisResample
arma::mat PhaseListAnalysisResample(arma::mat& x, const int& DIM_X, const int& DIM_Y, const int& SHUFFLES, const int& CORES);
RcppExport SEXP _InVivoR_PhaseListAnalysisResample(SEXP xSEXP, SEXP DIM_XSEXP, SEXP DIM_YSEXP, SEXP SHUFFLESSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int& >::type DIM_X(DIM_XSEXP);
    Rcpp::traits::input_parameter< const int& >::type DIM_Y(DIM_YSEXP);
    Rcpp::traits::input_parameter< const int& >::type SHUFFLES(SHUFFLESSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(PhaseListAnalysisResample(x, DIM_X, DIM_Y, SHUFFLES, CORES));
    return rcpp_result_gen;
END_RCPP
}
// StimulusSequence
Rcpp::List StimulusSequence(Rcpp::NumericVector& raw, Rcpp::NumericVector& filt, int& sampling_frequency, double& threshold, const double& max_time_gap);
RcppExport SEXP _InVivoR_StimulusSequence(SEXP rawSEXP, SEXP filtSEXP, SEXP sampling_frequencySEXP, SEXP thresholdSEXP, SEXP max_time_gapSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type raw(rawSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type filt(filtSEXP);
    Rcpp::traits::input_parameter< int& >::type sampling_frequency(sampling_frequencySEXP);
    Rcpp::traits::input_parameter< double& >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< const double& >::type max_time_gap(max_time_gapSEXP);
    rcpp_result_gen = Rcpp::wrap(StimulusSequence(raw, filt, sampling_frequency, threshold, max_time_gap));
    return rcpp_result_gen;
END_RCPP
}
// max_channel
arma::rowvec max_channel(arma::mat median_input_mat);
RcppExport SEXP _InVivoR_max_channel(SEXP median_input_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type median_input_mat(median_input_matSEXP);
    rcpp_result_gen = Rcpp::wrap(max_channel(median_input_mat));
    return rcpp_result_gen;
END_RCPP
}
// arma_apply_median
Rcpp::NumericVector arma_apply_median(arma::mat input_mat);
RcppExport SEXP _InVivoR_arma_apply_median(SEXP input_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type input_mat(input_matSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_apply_median(input_mat));
    return rcpp_result_gen;
END_RCPP
}
// arma_spike_extraction_loop
Rcpp::NumericVector arma_spike_extraction_loop(arma::vec input_mat_raw, int channel_nr);
RcppExport SEXP _InVivoR_arma_spike_extraction_loop(SEXP input_mat_rawSEXP, SEXP channel_nrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type input_mat_raw(input_mat_rawSEXP);
    Rcpp::traits::input_parameter< int >::type channel_nr(channel_nrSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_spike_extraction_loop(input_mat_raw, channel_nr));
    return rcpp_result_gen;
END_RCPP
}
// chan_out
arma::mat chan_out(Rcpp::List spike_shape_list, int channel_nr);
RcppExport SEXP _InVivoR_chan_out(SEXP spike_shape_listSEXP, SEXP channel_nrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type spike_shape_list(spike_shape_listSEXP);
    Rcpp::traits::input_parameter< int >::type channel_nr(channel_nrSEXP);
    rcpp_result_gen = Rcpp::wrap(chan_out(spike_shape_list, channel_nr));
    return rcpp_result_gen;
END_RCPP
}
// BWFilterCpp
arma::cx_vec BWFilterCpp(arma::cx_vec& InputFFT, const double& SamplingFrequency, const int& ORDER, const double& f0, const std::string type, const int& CORES);
RcppExport SEXP _InVivoR_BWFilterCpp(SEXP InputFFTSEXP, SEXP SamplingFrequencySEXP, SEXP ORDERSEXP, SEXP f0SEXP, SEXP typeSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cx_vec& >::type InputFFT(InputFFTSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFrequency(SamplingFrequencySEXP);
    Rcpp::traits::input_parameter< const int& >::type ORDER(ORDERSEXP);
    Rcpp::traits::input_parameter< const double& >::type f0(f0SEXP);
    Rcpp::traits::input_parameter< const std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(BWFilterCpp(InputFFT, SamplingFrequency, ORDER, f0, type, CORES));
    return rcpp_result_gen;
END_RCPP
}
// FirFiltering
Rcpp::NumericVector FirFiltering(const arma::colvec& SIGNAL, const arma::colvec& FIR_FILTER);
RcppExport SEXP _InVivoR_FirFiltering(SEXP SIGNALSEXP, SEXP FIR_FILTERSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type SIGNAL(SIGNALSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type FIR_FILTER(FIR_FILTERSEXP);
    rcpp_result_gen = Rcpp::wrap(FirFiltering(SIGNAL, FIR_FILTER));
    return rcpp_result_gen;
END_RCPP
}
// spike_ccf
Rcpp::IntegerVector spike_ccf(const Rcpp::NumericVector& x, const Rcpp::NumericVector& y, const double& WINDOW_LENGTH, const double& BIN_SIZE);
RcppExport SEXP _InVivoR_spike_ccf(SEXP xSEXP, SEXP ySEXP, SEXP WINDOW_LENGTHSEXP, SEXP BIN_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double& >::type WINDOW_LENGTH(WINDOW_LENGTHSEXP);
    Rcpp::traits::input_parameter< const double& >::type BIN_SIZE(BIN_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(spike_ccf(x, y, WINDOW_LENGTH, BIN_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// spike_ccf_batch
Rcpp::List spike_ccf_batch(const Rcpp::NumericVector& Time, const Rcpp::IntegerVector UnitNr, const double WINDOW_LENGTH, const double BIN_SIZE);
RcppExport SEXP _InVivoR_spike_ccf_batch(SEXP TimeSEXP, SEXP UnitNrSEXP, SEXP WINDOW_LENGTHSEXP, SEXP BIN_SIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type Time(TimeSEXP);
    Rcpp::traits::input_parameter< const Rcpp::IntegerVector >::type UnitNr(UnitNrSEXP);
    Rcpp::traits::input_parameter< const double >::type WINDOW_LENGTH(WINDOW_LENGTHSEXP);
    Rcpp::traits::input_parameter< const double >::type BIN_SIZE(BIN_SIZESEXP);
    rcpp_result_gen = Rcpp::wrap(spike_ccf_batch(Time, UnitNr, WINDOW_LENGTH, BIN_SIZE));
    return rcpp_result_gen;
END_RCPP
}
// ConfIntPoisson
Rcpp::NumericMatrix ConfIntPoisson(const arma::vec& CountVector, const double& CONFLEVEL, const double& SD, const double& CENTREMIN, const int& KERNELSIZE);
RcppExport SEXP _InVivoR_ConfIntPoisson(SEXP CountVectorSEXP, SEXP CONFLEVELSEXP, SEXP SDSEXP, SEXP CENTREMINSEXP, SEXP KERNELSIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type CountVector(CountVectorSEXP);
    Rcpp::traits::input_parameter< const double& >::type CONFLEVEL(CONFLEVELSEXP);
    Rcpp::traits::input_parameter< const double& >::type SD(SDSEXP);
    Rcpp::traits::input_parameter< const double& >::type CENTREMIN(CENTREMINSEXP);
    Rcpp::traits::input_parameter< const int& >::type KERNELSIZE(KERNELSIZESEXP);
    rcpp_result_gen = Rcpp::wrap(ConfIntPoisson(CountVector, CONFLEVEL, SD, CENTREMIN, KERNELSIZE));
    return rcpp_result_gen;
END_RCPP
}
// SpikeCCF
Rcpp::List SpikeCCF(Rcpp::NumericVector& x, Rcpp::NumericVector& y, const double& WINDOW_LENGTH, const double& BIN_SIZE, bool BaselineFrequency, bool ConfidenceInterval, double ConfLevel, const double& SD, const double& CENTREMIN, const int& KERNELSIZE);
RcppExport SEXP _InVivoR_SpikeCCF(SEXP xSEXP, SEXP ySEXP, SEXP WINDOW_LENGTHSEXP, SEXP BIN_SIZESEXP, SEXP BaselineFrequencySEXP, SEXP ConfidenceIntervalSEXP, SEXP ConfLevelSEXP, SEXP SDSEXP, SEXP CENTREMINSEXP, SEXP KERNELSIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const double& >::type WINDOW_LENGTH(WINDOW_LENGTHSEXP);
    Rcpp::traits::input_parameter< const double& >::type BIN_SIZE(BIN_SIZESEXP);
    Rcpp::traits::input_parameter< bool >::type BaselineFrequency(BaselineFrequencySEXP);
    Rcpp::traits::input_parameter< bool >::type ConfidenceInterval(ConfidenceIntervalSEXP);
    Rcpp::traits::input_parameter< double >::type ConfLevel(ConfLevelSEXP);
    Rcpp::traits::input_parameter< const double& >::type SD(SDSEXP);
    Rcpp::traits::input_parameter< const double& >::type CENTREMIN(CENTREMINSEXP);
    Rcpp::traits::input_parameter< const int& >::type KERNELSIZE(KERNELSIZESEXP);
    rcpp_result_gen = Rcpp::wrap(SpikeCCF(x, y, WINDOW_LENGTH, BIN_SIZE, BaselineFrequency, ConfidenceInterval, ConfLevel, SD, CENTREMIN, KERNELSIZE));
    return rcpp_result_gen;
END_RCPP
}
// spike_stim_properties
Rcpp::NumericMatrix spike_stim_properties(const Rcpp::IntegerVector& spike_idx, const Rcpp::NumericMatrix& stim_mat_org, const Rcpp::NumericMatrix& block_mat_org, const int& sampling_rate, const bool include_isolated);
RcppExport SEXP _InVivoR_spike_stim_properties(SEXP spike_idxSEXP, SEXP stim_mat_orgSEXP, SEXP block_mat_orgSEXP, SEXP sampling_rateSEXP, SEXP include_isolatedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::IntegerVector& >::type spike_idx(spike_idxSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type stim_mat_org(stim_mat_orgSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type block_mat_org(block_mat_orgSEXP);
    Rcpp::traits::input_parameter< const int& >::type sampling_rate(sampling_rateSEXP);
    Rcpp::traits::input_parameter< const bool >::type include_isolated(include_isolatedSEXP);
    rcpp_result_gen = Rcpp::wrap(spike_stim_properties(spike_idx, stim_mat_org, block_mat_org, sampling_rate, include_isolated));
    return rcpp_result_gen;
END_RCPP
}
// mat_baseline_zscore
arma::mat mat_baseline_zscore(arma::mat input_mat, int from_point, int to_point);
RcppExport SEXP _InVivoR_mat_baseline_zscore(SEXP input_matSEXP, SEXP from_pointSEXP, SEXP to_pointSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type input_mat(input_matSEXP);
    Rcpp::traits::input_parameter< int >::type from_point(from_pointSEXP);
    Rcpp::traits::input_parameter< int >::type to_point(to_pointSEXP);
    rcpp_result_gen = Rcpp::wrap(mat_baseline_zscore(input_mat, from_point, to_point));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_InVivoR_BinaryFileAccess", (DL_FUNC) &_InVivoR_BinaryFileAccess, 6},
    {"_InVivoR_decimate", (DL_FUNC) &_InVivoR_decimate, 3},
    {"_InVivoR_arma_gaussian", (DL_FUNC) &_InVivoR_arma_gaussian, 6},
    {"_InVivoR_arma_gaussian_kernel", (DL_FUNC) &_InVivoR_arma_gaussian_kernel, 3},
    {"_InVivoR_arma_gaussian_loop", (DL_FUNC) &_InVivoR_arma_gaussian_loop, 5},
    {"_InVivoR_BAKS", (DL_FUNC) &_InVivoR_BAKS, 4},
    {"_InVivoR_OKS", (DL_FUNC) &_InVivoR_OKS, 4},
    {"_InVivoR_BAKS_fast", (DL_FUNC) &_InVivoR_BAKS_fast, 4},
    {"_InVivoR_BAKS_fast_new", (DL_FUNC) &_InVivoR_BAKS_fast_new, 4},
    {"_InVivoR_MI", (DL_FUNC) &_InVivoR_MI, 10},
    {"_InVivoR_PhaseListAnalysis", (DL_FUNC) &_InVivoR_PhaseListAnalysis, 5},
    {"_InVivoR_PhaseListAnalysisShuffle", (DL_FUNC) &_InVivoR_PhaseListAnalysisShuffle, 5},
    {"_InVivoR_PhaseListAnalysisResample", (DL_FUNC) &_InVivoR_PhaseListAnalysisResample, 5},
    {"_InVivoR_StimulusSequence", (DL_FUNC) &_InVivoR_StimulusSequence, 5},
    {"_InVivoR_max_channel", (DL_FUNC) &_InVivoR_max_channel, 1},
    {"_InVivoR_arma_apply_median", (DL_FUNC) &_InVivoR_arma_apply_median, 1},
    {"_InVivoR_arma_spike_extraction_loop", (DL_FUNC) &_InVivoR_arma_spike_extraction_loop, 2},
    {"_InVivoR_chan_out", (DL_FUNC) &_InVivoR_chan_out, 2},
    {"_InVivoR_BWFilterCpp", (DL_FUNC) &_InVivoR_BWFilterCpp, 6},
    {"_InVivoR_FirFiltering", (DL_FUNC) &_InVivoR_FirFiltering, 2},
    {"_InVivoR_spike_ccf", (DL_FUNC) &_InVivoR_spike_ccf, 4},
    {"_InVivoR_spike_ccf_batch", (DL_FUNC) &_InVivoR_spike_ccf_batch, 4},
    {"_InVivoR_ConfIntPoisson", (DL_FUNC) &_InVivoR_ConfIntPoisson, 5},
    {"_InVivoR_SpikeCCF", (DL_FUNC) &_InVivoR_SpikeCCF, 10},
    {"_InVivoR_spike_stim_properties", (DL_FUNC) &_InVivoR_spike_stim_properties, 5},
    {"_InVivoR_mat_baseline_zscore", (DL_FUNC) &_InVivoR_mat_baseline_zscore, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_InVivoR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
