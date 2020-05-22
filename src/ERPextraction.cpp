#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' ERP (event-related potential) extraction
//' 
//' This function returns a matrix with extracted traces for any given range. 
//' The range is has to be provided in form of onset indeces and end indeces. 
//' Additional information can be provided to adjust for sampling differences 
//' between range points and trace sampling frequency. If required the window 
//' is proportionally elongated to include Pre/Post times.
//'
//' @param Trace A numeric vector which is used for extraction.
//' @param Onset An integer vector with onset sample indeces.
//' @param End An integer vector with end sample indeces.
//' @param SamplingFreqStim A double indicating the sampling frequency of onset/end indeces.
//' @param SamplingFreqTrace A double indicating the sampling frequency of the trace.
//' @param PrePhase A bool indicating if pre-onset timings should be included (default = false).
//' @param PostPhase A bool indicating if post-end timings should be included (default = false).
//' @return Returns a matrix with extracted ERPs.
//' @export
// [[Rcpp::export]]
arma::mat ERPMat(const arma::vec& Trace,
                 const arma::vec& Onset,
                 const arma::vec& End,
                 const double& SamplingFreqStim = 0,
                 const double& SamplingFreqTrace = 0,
                 const bool& PrePhase = false,
                 const bool& PostPhase = false) {
  arma::vec OnsetTmp(Onset.memptr(), Onset.n_elem);
  arma::vec EndTmp(End.memptr(), End.n_elem);
  if(SamplingFreqStim > 0 and SamplingFreqTrace > 0) {
    double SamplingFactor = SamplingFreqStim/SamplingFreqTrace;
    OnsetTmp /= SamplingFactor;
    EndTmp /= SamplingFactor;
  }
  arma::ivec OnsetNew = arma::conv_to<arma::ivec>::from(OnsetTmp);
  arma::ivec EndNew = arma::conv_to<arma::ivec>::from(EndTmp);
  int TraceLength = Trace.n_elem;
  int LengthFactor = 1;
  int PrePhaseInt = 0;
  if(PrePhase == true) {
    PrePhaseInt = 1;
    ++LengthFactor;
  } if(PostPhase == true) {
    ++LengthFactor;
  }
  // estimate length of stimulation block - caution: it is assumed that blocks have same length --> future implementation can 
  int ExtractionLength = median(EndNew-OnsetNew);
  arma::mat OutputMat = arma::mat(ExtractionLength*LengthFactor, OnsetNew.n_elem, arma::fill::zeros);
  int windowLength = ExtractionLength*LengthFactor-1;
  for(unsigned int i = 0; i < OnsetNew.n_elem; ++i) {
    int StartInt = OnsetNew.at(i)-ExtractionLength*PrePhaseInt;
    int EndInt = StartInt + windowLength;
    // bounds check
    if(EndInt < TraceLength and StartInt >= 0) {
      // subvec implementation
      OutputMat.col(i) = Trace.subvec(StartInt,EndInt);
    } else if(EndInt > TraceLength and StartInt >= 0) {
      // start is later than 0
      for(int vecIdx = 0; vecIdx < windowLength; ++vecIdx) {
        // insert elements into gaps
        OutputMat.at(vecIdx,i) = Trace.at(StartInt);
        ++StartInt;
        if(StartInt==TraceLength) {
          break;
        }
      }
    } else if(StartInt < 0 and EndInt < TraceLength) {
      // insert elements starting after offset
      int Offset = 0-StartInt;
      StartInt = 0;
      for(int vecIdx = Offset; vecIdx < windowLength; ++vecIdx) {
        // insert elements into gaps
        OutputMat.at(vecIdx,i) = Trace.at(StartInt);
        ++StartInt;
      }
    } else {
      Rcpp::warning("Range error");
    }
  }
  return OutputMat.t();
}

//' Wrapper for ERP (event-related potential) extraction
//' 
//' This function returns a list which includes the frequencies and matrices 
//' with extracted traces for any given range. The range is has to be provided
//' in form of onset indeces and end indeces. Additional information can be 
//' provided to adjust for sampling differences between range points and trace 
//' sampling frequency. If required the window is proportionally elongated to 
//' include Pre/Post times.
//'
//' @param Trace A numeric vector which is used for extraction.
//' @param BlockMat The stimulation block matrix from StimulusSequence() including onset and end timings.
//' @param SamplingFreqStim A double indicating the sampling frequency of onset/end indeces.
//' @param SamplingFreqTrace A double indicating the sampling frequency of the trace.
//' @param PrePhase A bool indicating if pre-onset timings should be included (default = true).
//' @param PostPhase A bool indicating if post-end timings should be included (default = true).
//' @param FixStartLength A double indicating the starting length when fixed window is used (default = 0).
//' @param WindowLength A double indicating the time block in seconds (default = 0).
//' @return Returns a list which includes stimulation frequencies for hyperblocks and the corresponding ERP matrices.
//' @export
// [[Rcpp::export]]
Rcpp::List ERPList(const arma::vec& Trace,
                   const arma::mat BlockMat,
                   const double& SamplingFreqStim = 0,
                   const double& SamplingFreqTrace = 0,
                   bool PrePhase = true,
                   bool PostPhase = true,
                   const double& FixStartLength = 0,
                   const double& WindowLength = 0) {
  Rcpp::StringVector BlockProperties(BlockMat.n_rows);
  arma::vec OnsetCol = BlockMat.col(0);
  arma::vec Onset = arma::conv_to<arma::vec>::from(OnsetCol);
  arma::vec EndCol = BlockMat.col(1);
  arma::vec End = arma::conv_to<arma::vec>::from(EndCol);
  if(FixStartLength > 0 and WindowLength > 0) {
    if(SamplingFreqStim == 0) {
      Rcpp::stop("Sampling frequency missing");
    }
    Onset -= SamplingFreqStim*FixStartLength;
    End = Onset+SamplingFreqStim*WindowLength;
    PrePhase = false;
    PostPhase = false;
  }
  for(int i = 0; i < BlockProperties.size(); ++i) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << BlockMat.at(i,15);
    std::string PropString = stream.str();
    if(BlockMat.at(i,10) == 1) {
      PropString += "_sine";
    } else if(BlockMat.at(i,11) == 1) {
      PropString += "_pulse";
    }
    if(BlockMat.at(i,7) == 1) {
      PropString += "_burst_";
      std::stringstream burst;
      burst << std::fixed << std::setprecision(1) << BlockMat.at(i,9);
      PropString += burst.str();
    }
    if(BlockMat.at(i,12) == 1) {
      PropString += "_frontRamp";
    }
    if(BlockMat.at(i,13) == 1) {
      PropString += "_frontEnd";
    }
    BlockProperties[i] = PropString;
  }
  Rcpp::StringVector UniqueProp = Rcpp::unique(BlockProperties);
  Rcpp::List ListERP(UniqueProp.size());
  for(int i = 0; i < UniqueProp.size(); ++i) {
    arma::uvec Idx = arma::uvec (BlockProperties.size(), arma::fill::zeros);
    int k = 0;
    for(int j = 0; j < BlockProperties.size(); ++j) {
      if(BlockProperties[j]==UniqueProp[i]) {
        Idx.at(k) = j;
        ++k;
      }
    }
    Idx.resize(k);
    arma::vec OnsetTmp = Onset.elem(Idx);
    arma::vec EndTmp = End.elem(Idx);
    ListERP[i] = ERPMat(Trace, OnsetTmp, EndTmp, SamplingFreqStim, SamplingFreqTrace, PrePhase, PostPhase);
  }
  return Rcpp::List::create(Rcpp::Named("ProtocolNames") = UniqueProp,
                            Rcpp::Named("ERP") = ListERP);
}