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
// StimFileRead
Rcpp::NumericVector StimFileRead(const std::string& FILENAME, const bool digital);
RcppExport SEXP _InVivoR_StimFileRead(SEXP FILENAMESEXP, SEXP digitalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type FILENAME(FILENAMESEXP);
    Rcpp::traits::input_parameter< const bool >::type digital(digitalSEXP);
    rcpp_result_gen = Rcpp::wrap(StimFileRead(FILENAME, digital));
    return rcpp_result_gen;
END_RCPP
}
// AmpFileRead
Rcpp::NumericMatrix AmpFileRead(const std::string& FILENAME, const int ChannelNumber);
RcppExport SEXP _InVivoR_AmpFileRead(SEXP FILENAMESEXP, SEXP ChannelNumberSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type FILENAME(FILENAMESEXP);
    Rcpp::traits::input_parameter< const int >::type ChannelNumber(ChannelNumberSEXP);
    rcpp_result_gen = Rcpp::wrap(AmpFileRead(FILENAME, ChannelNumber));
    return rcpp_result_gen;
END_RCPP
}
// AmpFileReadMerge
arma::mat AmpFileReadMerge(const std::string& FILENAME1, const std::string& FILENAME2, const int ChannelNumber);
RcppExport SEXP _InVivoR_AmpFileReadMerge(SEXP FILENAME1SEXP, SEXP FILENAME2SEXP, SEXP ChannelNumberSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type FILENAME1(FILENAME1SEXP);
    Rcpp::traits::input_parameter< const std::string& >::type FILENAME2(FILENAME2SEXP);
    Rcpp::traits::input_parameter< const int >::type ChannelNumber(ChannelNumberSEXP);
    rcpp_result_gen = Rcpp::wrap(AmpFileReadMerge(FILENAME1, FILENAME2, ChannelNumber));
    return rcpp_result_gen;
END_RCPP
}
// convertToBinary
Rcpp::List convertToBinary(arma::vec x);
RcppExport SEXP _InVivoR_convertToBinary(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(convertToBinary(x));
    return rcpp_result_gen;
END_RCPP
}
// decimate
Rcpp::NumericVector decimate(const arma::vec& SIGNAL, const arma::vec& FIR_FILTER, const int& M);
RcppExport SEXP _InVivoR_decimate(SEXP SIGNALSEXP, SEXP FIR_FILTERSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type SIGNAL(SIGNALSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type FIR_FILTER(FIR_FILTERSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(decimate(SIGNAL, FIR_FILTER, M));
    return rcpp_result_gen;
END_RCPP
}
// ERPMat
arma::mat ERPMat(const arma::vec& Trace, const arma::vec& Onset, const arma::vec& End, const double& SamplingFreqStim, const double& SamplingFreqTrace, const bool& PrePhase, const bool& PostPhase);
RcppExport SEXP _InVivoR_ERPMat(SEXP TraceSEXP, SEXP OnsetSEXP, SEXP EndSEXP, SEXP SamplingFreqStimSEXP, SEXP SamplingFreqTraceSEXP, SEXP PrePhaseSEXP, SEXP PostPhaseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type Trace(TraceSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Onset(OnsetSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type End(EndSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFreqStim(SamplingFreqStimSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFreqTrace(SamplingFreqTraceSEXP);
    Rcpp::traits::input_parameter< const bool& >::type PrePhase(PrePhaseSEXP);
    Rcpp::traits::input_parameter< const bool& >::type PostPhase(PostPhaseSEXP);
    rcpp_result_gen = Rcpp::wrap(ERPMat(Trace, Onset, End, SamplingFreqStim, SamplingFreqTrace, PrePhase, PostPhase));
    return rcpp_result_gen;
END_RCPP
}
// ERPList
Rcpp::List ERPList(const arma::vec& Trace, const arma::mat BlockMat, const double& SamplingFreqStim, const double& SamplingFreqTrace, bool PrePhase, bool PostPhase, const double& FixStartLength, const double& WindowLength);
RcppExport SEXP _InVivoR_ERPList(SEXP TraceSEXP, SEXP BlockMatSEXP, SEXP SamplingFreqStimSEXP, SEXP SamplingFreqTraceSEXP, SEXP PrePhaseSEXP, SEXP PostPhaseSEXP, SEXP FixStartLengthSEXP, SEXP WindowLengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type Trace(TraceSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type BlockMat(BlockMatSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFreqStim(SamplingFreqStimSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFreqTrace(SamplingFreqTraceSEXP);
    Rcpp::traits::input_parameter< bool >::type PrePhase(PrePhaseSEXP);
    Rcpp::traits::input_parameter< bool >::type PostPhase(PostPhaseSEXP);
    Rcpp::traits::input_parameter< const double& >::type FixStartLength(FixStartLengthSEXP);
    Rcpp::traits::input_parameter< const double& >::type WindowLength(WindowLengthSEXP);
    rcpp_result_gen = Rcpp::wrap(ERPList(Trace, BlockMat, SamplingFreqStim, SamplingFreqTrace, PrePhase, PostPhase, FixStartLength, WindowLength));
    return rcpp_result_gen;
END_RCPP
}
// FiringRate
Rcpp::NumericVector FiringRate(arma::vec& SpikeTimes, Rcpp::Nullable<double> timeStart, Rcpp::Nullable<double> timeEnd, Rcpp::Nullable<double> sigma, Rcpp::Nullable<double> alpha, bool useBAKS, double BAKSalpha, double BAKSbeta, double SamplingRate, int CORES);
RcppExport SEXP _InVivoR_FiringRate(SEXP SpikeTimesSEXP, SEXP timeStartSEXP, SEXP timeEndSEXP, SEXP sigmaSEXP, SEXP alphaSEXP, SEXP useBAKSSEXP, SEXP BAKSalphaSEXP, SEXP BAKSbetaSEXP, SEXP SamplingRateSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimes(SpikeTimesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type timeStart(timeStartSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type timeEnd(timeEndSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< bool >::type useBAKS(useBAKSSEXP);
    Rcpp::traits::input_parameter< double >::type BAKSalpha(BAKSalphaSEXP);
    Rcpp::traits::input_parameter< double >::type BAKSbeta(BAKSbetaSEXP);
    Rcpp::traits::input_parameter< double >::type SamplingRate(SamplingRateSEXP);
    Rcpp::traits::input_parameter< int >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(FiringRate(SpikeTimes, timeStart, timeEnd, sigma, alpha, useBAKS, BAKSalpha, BAKSbeta, SamplingRate, CORES));
    return rcpp_result_gen;
END_RCPP
}
// FiringRateSparse
Rcpp::NumericVector FiringRateSparse(arma::vec& SpikeTimes, Rcpp::Nullable<double> sigma, Rcpp::Nullable<double> alpha, double SamplingRate);
RcppExport SEXP _InVivoR_FiringRateSparse(SEXP SpikeTimesSEXP, SEXP sigmaSEXP, SEXP alphaSEXP, SEXP SamplingRateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec& >::type SpikeTimes(SpikeTimesSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<double> >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type SamplingRate(SamplingRateSEXP);
    rcpp_result_gen = Rcpp::wrap(FiringRateSparse(SpikeTimes, sigma, alpha, SamplingRate));
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
// FirFilteringOverlap
Rcpp::NumericVector FirFilteringOverlap(const arma::colvec& SIGNAL, const arma::colvec& FIR_FILTER, bool FiltFilt, unsigned int BatchSize, const int& CORES);
RcppExport SEXP _InVivoR_FirFilteringOverlap(SEXP SIGNALSEXP, SEXP FIR_FILTERSEXP, SEXP FiltFiltSEXP, SEXP BatchSizeSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type SIGNAL(SIGNALSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type FIR_FILTER(FIR_FILTERSEXP);
    Rcpp::traits::input_parameter< bool >::type FiltFilt(FiltFiltSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type BatchSize(BatchSizeSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(FirFilteringOverlap(SIGNAL, FIR_FILTER, FiltFilt, BatchSize, CORES));
    return rcpp_result_gen;
END_RCPP
}
// BWFiltCppOverlap
Rcpp::NumericVector BWFiltCppOverlap(const arma::vec& InputSignal, const double& SamplingFrequency, const int& ORDER, const double& f0, const std::string type, int BatchSize, const int& CORES);
RcppExport SEXP _InVivoR_BWFiltCppOverlap(SEXP InputSignalSEXP, SEXP SamplingFrequencySEXP, SEXP ORDERSEXP, SEXP f0SEXP, SEXP typeSEXP, SEXP BatchSizeSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type InputSignal(InputSignalSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFrequency(SamplingFrequencySEXP);
    Rcpp::traits::input_parameter< const int& >::type ORDER(ORDERSEXP);
    Rcpp::traits::input_parameter< const double& >::type f0(f0SEXP);
    Rcpp::traits::input_parameter< const std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type BatchSize(BatchSizeSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(BWFiltCppOverlap(InputSignal, SamplingFrequency, ORDER, f0, type, BatchSize, CORES));
    return rcpp_result_gen;
END_RCPP
}
// PhaseListAnalysis
Rcpp::List PhaseListAnalysis(const arma::cube& x, const int& CORES);
RcppExport SEXP _InVivoR_PhaseListAnalysis(SEXP xSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(PhaseListAnalysis(x, CORES));
    return rcpp_result_gen;
END_RCPP
}
// PhaseListAnalysisShuffle
arma::mat PhaseListAnalysisShuffle(arma::mat& x, const int SHUFFLES, int CORES);
RcppExport SEXP _InVivoR_PhaseListAnalysisShuffle(SEXP xSEXP, SEXP SHUFFLESSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int >::type SHUFFLES(SHUFFLESSEXP);
    Rcpp::traits::input_parameter< int >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(PhaseListAnalysisShuffle(x, SHUFFLES, CORES));
    return rcpp_result_gen;
END_RCPP
}
// PhaseListAnalysisResample
arma::mat PhaseListAnalysisResample(arma::mat& x, const int SHUFFLES, const int CORES);
RcppExport SEXP _InVivoR_PhaseListAnalysisResample(SEXP xSEXP, SEXP SHUFFLESSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const int >::type SHUFFLES(SHUFFLESSEXP);
    Rcpp::traits::input_parameter< const int >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(PhaseListAnalysisResample(x, SHUFFLES, CORES));
    return rcpp_result_gen;
END_RCPP
}
// BWFiltCpp
Rcpp::NumericVector BWFiltCpp(const arma::vec& InputSignal, const double& SamplingFrequency, const int& ORDER, const double& f0, const std::string type, int BatchSize, const int& CORES);
RcppExport SEXP _InVivoR_BWFiltCpp(SEXP InputSignalSEXP, SEXP SamplingFrequencySEXP, SEXP ORDERSEXP, SEXP f0SEXP, SEXP typeSEXP, SEXP BatchSizeSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type InputSignal(InputSignalSEXP);
    Rcpp::traits::input_parameter< const double& >::type SamplingFrequency(SamplingFrequencySEXP);
    Rcpp::traits::input_parameter< const int& >::type ORDER(ORDERSEXP);
    Rcpp::traits::input_parameter< const double& >::type f0(f0SEXP);
    Rcpp::traits::input_parameter< const std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type BatchSize(BatchSizeSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(BWFiltCpp(InputSignal, SamplingFrequency, ORDER, f0, type, BatchSize, CORES));
    return rcpp_result_gen;
END_RCPP
}
// StimulusSequence
Rcpp::List StimulusSequence(Rcpp::NumericVector& raw, int& sampling_frequency, double& threshold, const double& max_time_gap, const int CORES);
RcppExport SEXP _InVivoR_StimulusSequence(SEXP rawSEXP, SEXP sampling_frequencySEXP, SEXP thresholdSEXP, SEXP max_time_gapSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector& >::type raw(rawSEXP);
    Rcpp::traits::input_parameter< int& >::type sampling_frequency(sampling_frequencySEXP);
    Rcpp::traits::input_parameter< double& >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< const double& >::type max_time_gap(max_time_gapSEXP);
    Rcpp::traits::input_parameter< const int >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(StimulusSequence(raw, sampling_frequency, threshold, max_time_gap, CORES));
    return rcpp_result_gen;
END_RCPP
}
// WT
arma::cx_mat WT(const arma::vec& Signal, const arma::vec& frequencies, const double& samplingfrequency, const double& sigma, const double& LNorm, int CORES);
RcppExport SEXP _InVivoR_WT(SEXP SignalSEXP, SEXP frequenciesSEXP, SEXP samplingfrequencySEXP, SEXP sigmaSEXP, SEXP LNormSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type Signal(SignalSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type frequencies(frequenciesSEXP);
    Rcpp::traits::input_parameter< const double& >::type samplingfrequency(samplingfrequencySEXP);
    Rcpp::traits::input_parameter< const double& >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< const double& >::type LNorm(LNormSEXP);
    Rcpp::traits::input_parameter< int >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(WT(Signal, frequencies, samplingfrequency, sigma, LNorm, CORES));
    return rcpp_result_gen;
END_RCPP
}
// WTbatch
Rcpp::List WTbatch(arma::mat ERPMat, const arma::vec& frequencies, const double& samplingfrequency, const double& sigma, const double& LNorm, int CORES, bool compression, bool PhaseAnalysis);
RcppExport SEXP _InVivoR_WTbatch(SEXP ERPMatSEXP, SEXP frequenciesSEXP, SEXP samplingfrequencySEXP, SEXP sigmaSEXP, SEXP LNormSEXP, SEXP CORESSEXP, SEXP compressionSEXP, SEXP PhaseAnalysisSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type ERPMat(ERPMatSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type frequencies(frequenciesSEXP);
    Rcpp::traits::input_parameter< const double& >::type samplingfrequency(samplingfrequencySEXP);
    Rcpp::traits::input_parameter< const double& >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< const double& >::type LNorm(LNormSEXP);
    Rcpp::traits::input_parameter< int >::type CORES(CORESSEXP);
    Rcpp::traits::input_parameter< bool >::type compression(compressionSEXP);
    Rcpp::traits::input_parameter< bool >::type PhaseAnalysis(PhaseAnalysisSEXP);
    rcpp_result_gen = Rcpp::wrap(WTbatch(ERPMat, frequencies, samplingfrequency, sigma, LNorm, CORES, compression, PhaseAnalysis));
    return rcpp_result_gen;
END_RCPP
}
// PowerMat
arma::mat PowerMat(const arma::cube& x, const bool& ZScore);
RcppExport SEXP _InVivoR_PowerMat(SEXP xSEXP, SEXP ZScoreSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const bool& >::type ZScore(ZScoreSEXP);
    rcpp_result_gen = Rcpp::wrap(PowerMat(x, ZScore));
    return rcpp_result_gen;
END_RCPP
}
// WTSqueeze
arma::cx_mat WTSqueeze(const arma::cx_mat& WT, const arma::vec& frequencies, const double& samplingfrequency, double sigma, const int& CORES);
RcppExport SEXP _InVivoR_WTSqueeze(SEXP WTSEXP, SEXP frequenciesSEXP, SEXP samplingfrequencySEXP, SEXP sigmaSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cx_mat& >::type WT(WTSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type frequencies(frequenciesSEXP);
    Rcpp::traits::input_parameter< const double& >::type samplingfrequency(samplingfrequencySEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(WTSqueeze(WT, frequencies, samplingfrequency, sigma, CORES));
    return rcpp_result_gen;
END_RCPP
}
// CxCubeCollapse
arma::cx_mat CxCubeCollapse(const arma::cx_cube& x);
RcppExport SEXP _InVivoR_CxCubeCollapse(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cx_cube& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(CxCubeCollapse(x));
    return rcpp_result_gen;
END_RCPP
}
// WTCoherence
Rcpp::List WTCoherence(arma::cx_mat& WT1, arma::cx_mat& WT2, arma::vec& frequencies, const double& samplingfrequency, const double& tKernelWidth, const double& sKernelWidth);
RcppExport SEXP _InVivoR_WTCoherence(SEXP WT1SEXP, SEXP WT2SEXP, SEXP frequenciesSEXP, SEXP samplingfrequencySEXP, SEXP tKernelWidthSEXP, SEXP sKernelWidthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cx_mat& >::type WT1(WT1SEXP);
    Rcpp::traits::input_parameter< arma::cx_mat& >::type WT2(WT2SEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type frequencies(frequenciesSEXP);
    Rcpp::traits::input_parameter< const double& >::type samplingfrequency(samplingfrequencySEXP);
    Rcpp::traits::input_parameter< const double& >::type tKernelWidth(tKernelWidthSEXP);
    Rcpp::traits::input_parameter< const double& >::type sKernelWidth(sKernelWidthSEXP);
    rcpp_result_gen = Rcpp::wrap(WTCoherence(WT1, WT2, frequencies, samplingfrequency, tKernelWidth, sKernelWidth));
    return rcpp_result_gen;
END_RCPP
}
// SpikeCut
arma::cube SpikeCut(const arma::mat& AmpMatrix, const arma::vec& SpikeIdx, const int& WINDOW);
RcppExport SEXP _InVivoR_SpikeCut(SEXP AmpMatrixSEXP, SEXP SpikeIdxSEXP, SEXP WINDOWSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type AmpMatrix(AmpMatrixSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type SpikeIdx(SpikeIdxSEXP);
    Rcpp::traits::input_parameter< const int& >::type WINDOW(WINDOWSEXP);
    rcpp_result_gen = Rcpp::wrap(SpikeCut(AmpMatrix, SpikeIdx, WINDOW));
    return rcpp_result_gen;
END_RCPP
}
// SpikeMed
arma::mat SpikeMed(const arma::cube& SpikeCube);
RcppExport SEXP _InVivoR_SpikeMed(SEXP SpikeCubeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type SpikeCube(SpikeCubeSEXP);
    rcpp_result_gen = Rcpp::wrap(SpikeMed(SpikeCube));
    return rcpp_result_gen;
END_RCPP
}
// MaxChannel
arma::rowvec MaxChannel(const arma::mat& MedianSpikeMat);
RcppExport SEXP _InVivoR_MaxChannel(SEXP MedianSpikeMatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type MedianSpikeMat(MedianSpikeMatSEXP);
    rcpp_result_gen = Rcpp::wrap(MaxChannel(MedianSpikeMat));
    return rcpp_result_gen;
END_RCPP
}
// ChannelFromList
Rcpp::List ChannelFromList(const Rcpp::List& SpikeCubeList);
RcppExport SEXP _InVivoR_ChannelFromList(SEXP SpikeCubeListSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::List& >::type SpikeCubeList(SpikeCubeListSEXP);
    rcpp_result_gen = Rcpp::wrap(ChannelFromList(SpikeCubeList));
    return rcpp_result_gen;
END_RCPP
}
// UnitChannel
Rcpp::List UnitChannel(const arma::vec& SpikeIdx, const arma::vec& Units, const arma::mat& AmpMatrix, const int& WINDOW);
RcppExport SEXP _InVivoR_UnitChannel(SEXP SpikeIdxSEXP, SEXP UnitsSEXP, SEXP AmpMatrixSEXP, SEXP WINDOWSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type SpikeIdx(SpikeIdxSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type Units(UnitsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type AmpMatrix(AmpMatrixSEXP);
    Rcpp::traits::input_parameter< const int& >::type WINDOW(WINDOWSEXP);
    rcpp_result_gen = Rcpp::wrap(UnitChannel(SpikeIdx, Units, AmpMatrix, WINDOW));
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
Rcpp::NumericVector FirFiltering(const arma::colvec& SIGNAL, const arma::colvec& FIR_FILTER, bool FiltFilt, unsigned int BatchSize, const int& CORES);
RcppExport SEXP _InVivoR_FirFiltering(SEXP SIGNALSEXP, SEXP FIR_FILTERSEXP, SEXP FiltFiltSEXP, SEXP BatchSizeSEXP, SEXP CORESSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type SIGNAL(SIGNALSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type FIR_FILTER(FIR_FILTERSEXP);
    Rcpp::traits::input_parameter< bool >::type FiltFilt(FiltFiltSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type BatchSize(BatchSizeSEXP);
    Rcpp::traits::input_parameter< const int& >::type CORES(CORESSEXP);
    rcpp_result_gen = Rcpp::wrap(FirFiltering(SIGNAL, FIR_FILTER, FiltFilt, BatchSize, CORES));
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
Rcpp::List SpikeCCF(const Rcpp::NumericVector x, Rcpp::Nullable<Rcpp::NumericVector> y, Rcpp::Nullable<Rcpp::IntegerVector> UnitNr, const double WINDOW_LENGTH, const double BIN_SIZE, const bool BaselineFrequency, const bool ConfidenceInterval, const double ConfLevel, const double SD, const double CENTREMIN, const int KERNELSIZE);
RcppExport SEXP _InVivoR_SpikeCCF(SEXP xSEXP, SEXP ySEXP, SEXP UnitNrSEXP, SEXP WINDOW_LENGTHSEXP, SEXP BIN_SIZESEXP, SEXP BaselineFrequencySEXP, SEXP ConfidenceIntervalSEXP, SEXP ConfLevelSEXP, SEXP SDSEXP, SEXP CENTREMINSEXP, SEXP KERNELSIZESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::NumericVector> >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::IntegerVector> >::type UnitNr(UnitNrSEXP);
    Rcpp::traits::input_parameter< const double >::type WINDOW_LENGTH(WINDOW_LENGTHSEXP);
    Rcpp::traits::input_parameter< const double >::type BIN_SIZE(BIN_SIZESEXP);
    Rcpp::traits::input_parameter< const bool >::type BaselineFrequency(BaselineFrequencySEXP);
    Rcpp::traits::input_parameter< const bool >::type ConfidenceInterval(ConfidenceIntervalSEXP);
    Rcpp::traits::input_parameter< const double >::type ConfLevel(ConfLevelSEXP);
    Rcpp::traits::input_parameter< const double >::type SD(SDSEXP);
    Rcpp::traits::input_parameter< const double >::type CENTREMIN(CENTREMINSEXP);
    Rcpp::traits::input_parameter< const int >::type KERNELSIZE(KERNELSIZESEXP);
    rcpp_result_gen = Rcpp::wrap(SpikeCCF(x, y, UnitNr, WINDOW_LENGTH, BIN_SIZE, BaselineFrequency, ConfidenceInterval, ConfLevel, SD, CENTREMIN, KERNELSIZE));
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
    {"_InVivoR_StimFileRead", (DL_FUNC) &_InVivoR_StimFileRead, 2},
    {"_InVivoR_AmpFileRead", (DL_FUNC) &_InVivoR_AmpFileRead, 2},
    {"_InVivoR_AmpFileReadMerge", (DL_FUNC) &_InVivoR_AmpFileReadMerge, 3},
    {"_InVivoR_convertToBinary", (DL_FUNC) &_InVivoR_convertToBinary, 1},
    {"_InVivoR_decimate", (DL_FUNC) &_InVivoR_decimate, 3},
    {"_InVivoR_ERPMat", (DL_FUNC) &_InVivoR_ERPMat, 7},
    {"_InVivoR_ERPList", (DL_FUNC) &_InVivoR_ERPList, 8},
    {"_InVivoR_FiringRate", (DL_FUNC) &_InVivoR_FiringRate, 10},
    {"_InVivoR_FiringRateSparse", (DL_FUNC) &_InVivoR_FiringRateSparse, 4},
    {"_InVivoR_MI", (DL_FUNC) &_InVivoR_MI, 10},
    {"_InVivoR_FirFilteringOverlap", (DL_FUNC) &_InVivoR_FirFilteringOverlap, 5},
    {"_InVivoR_BWFiltCppOverlap", (DL_FUNC) &_InVivoR_BWFiltCppOverlap, 7},
    {"_InVivoR_PhaseListAnalysis", (DL_FUNC) &_InVivoR_PhaseListAnalysis, 2},
    {"_InVivoR_PhaseListAnalysisShuffle", (DL_FUNC) &_InVivoR_PhaseListAnalysisShuffle, 3},
    {"_InVivoR_PhaseListAnalysisResample", (DL_FUNC) &_InVivoR_PhaseListAnalysisResample, 3},
    {"_InVivoR_BWFiltCpp", (DL_FUNC) &_InVivoR_BWFiltCpp, 7},
    {"_InVivoR_StimulusSequence", (DL_FUNC) &_InVivoR_StimulusSequence, 5},
    {"_InVivoR_WT", (DL_FUNC) &_InVivoR_WT, 6},
    {"_InVivoR_WTbatch", (DL_FUNC) &_InVivoR_WTbatch, 8},
    {"_InVivoR_PowerMat", (DL_FUNC) &_InVivoR_PowerMat, 2},
    {"_InVivoR_WTSqueeze", (DL_FUNC) &_InVivoR_WTSqueeze, 5},
    {"_InVivoR_CxCubeCollapse", (DL_FUNC) &_InVivoR_CxCubeCollapse, 1},
    {"_InVivoR_WTCoherence", (DL_FUNC) &_InVivoR_WTCoherence, 6},
    {"_InVivoR_SpikeCut", (DL_FUNC) &_InVivoR_SpikeCut, 3},
    {"_InVivoR_SpikeMed", (DL_FUNC) &_InVivoR_SpikeMed, 1},
    {"_InVivoR_MaxChannel", (DL_FUNC) &_InVivoR_MaxChannel, 1},
    {"_InVivoR_ChannelFromList", (DL_FUNC) &_InVivoR_ChannelFromList, 1},
    {"_InVivoR_UnitChannel", (DL_FUNC) &_InVivoR_UnitChannel, 4},
    {"_InVivoR_BWFilterCpp", (DL_FUNC) &_InVivoR_BWFilterCpp, 6},
    {"_InVivoR_FirFiltering", (DL_FUNC) &_InVivoR_FirFiltering, 5},
    {"_InVivoR_ConfIntPoisson", (DL_FUNC) &_InVivoR_ConfIntPoisson, 5},
    {"_InVivoR_SpikeCCF", (DL_FUNC) &_InVivoR_SpikeCCF, 11},
    {"_InVivoR_spike_stim_properties", (DL_FUNC) &_InVivoR_spike_stim_properties, 5},
    {"_InVivoR_mat_baseline_zscore", (DL_FUNC) &_InVivoR_mat_baseline_zscore, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_InVivoR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
