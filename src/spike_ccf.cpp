//#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

Rcpp::IntegerVector spike_ccf(const Rcpp::NumericVector&  x,
                              const Rcpp::NumericVector&  y,
                              const double& WINDOW_LENGTH = 1,
                              const double& BIN_SIZE = 0.001) {
  int l_x = x.size(),
    l_y = y.size();
  bool remainder_check = false;
  double total_width = roundf(WINDOW_LENGTH/BIN_SIZE)+1,
    diff_measure =  0,
    left_end = -WINDOW_LENGTH/2-BIN_SIZE/2,
    right_end = WINDOW_LENGTH/2+BIN_SIZE/2,
    out_id_double = 0;
  int half_w = (total_width-1)/2,
    last_j = 0, j = 0,
    loop_active = 1,
    out_id = 0,
    tmp_out = 0;
  Rcpp::IntegerVector hist_out(total_width);
  if(l_x == l_y and is_true(all(x==y))){
    for (int i = 0; i < l_y; ++i) {
      // inner while loop and check for window
      j = last_j;
      loop_active = 1;
      while (loop_active == 1 && j < l_x) {
        if(i!=j) {
          //calculate differences
          diff_measure = x[j]-y[i];
          if(diff_measure < left_end) {
            //find last valid point for next iteration
            last_j = j;
          } else if(diff_measure <= right_end && diff_measure >= left_end) {
            //calculate bin
            out_id_double = diff_measure/BIN_SIZE+half_w;
            out_id = out_id_double;
            tmp_out = out_id_double*2;
            remainder_check = (tmp_out % 2L == 1);
            //remainder_check to round values faster than roundf()
            if(remainder_check == 1) {
              ++out_id;
            }
            ++hist_out[out_id];
          } else if(diff_measure > right_end) {
            //exit while loop when out of window
            loop_active = 0;
          }
          ++j;
        } else {
          ++j;
        }
      }
    }
  } else {
    for (int i = 0; i < l_y; ++i) {
      // inner while loop and check for window
      j = last_j;
      loop_active = 1;
      while (loop_active == 1 && j < l_x) {
        diff_measure = x[j]-y[i];
        if(diff_measure < left_end) {
          //find last valid point for next iteration
          last_j = j;
        } else if(diff_measure <= right_end && diff_measure >= left_end) {
          //calculate bin
          out_id_double = diff_measure/BIN_SIZE+half_w;
          out_id = out_id_double;
          tmp_out = out_id_double*2;
          remainder_check = (tmp_out % 2L == 1);
          //remainder_check to round values faster than roundf()
          if(remainder_check == 1) {
            ++out_id;
          }
          ++hist_out[out_id];
        } else if(diff_measure > right_end ||j == l_x) {
          //exit while loop when out of window or finished
          loop_active = 0;
        }
        ++j;
      }
    }
  }
  //return integer vector
  return hist_out;
}

//' @title Confidence interval for poisson train
//' 
//' Computes the confidence interval of a poisson train using an inverted gaussian
//' distribution. Based on the weighted average with the given distribution lamda
//' can be estimated. The confidence interval is then estimated by estimation through
//' Chisq-distribution.
//'
//' @name ConfIntPoisson
//' @param CountVector An integer vector of counts estimated by Spike CCF.
//' @param CONFLEVEL A double indicating the confidence-level (default = 0.95).
//' @param SD A double as standard deviation for a gaussian shape parameter (default = 0.6).
//' @param CENTREMIN A double as shape parameter determining the strength of centre exclusion (default = 0.6).
//' @param KERNELSIZE A double as length parameter for gaussian kernel (2*KERNELSIZE+1, default = 20).
//' 
//' @return Returns a list containing counts per bin, axis, random bin count, confidence-intervals with counts per bin.
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix ConfIntPoisson(const arma::vec& CountVector,
                                   const double& CONFLEVEL = 0.95,
                                   const double& SD = 0.6,
                                   const double& CENTREMIN = 0.6,
                                   const int& KERNELSIZE = 20) {
  const double& alpha = 1-CONFLEVEL;
  const int& endPoint = (KERNELSIZE+CountVector.size());
  const double& LowerBound = alpha/2;
  const double& UpperBound = 1-alpha/2;
  arma::vec GaussKernel = arma::normpdf(arma::linspace(-KERNELSIZE, KERNELSIZE, 2*KERNELSIZE+1), 0, SD);
  GaussKernel = GaussKernel/arma::max(GaussKernel);
  GaussKernel = 1-(GaussKernel*(1-CENTREMIN));
  GaussKernel = GaussKernel/arma::mean(GaussKernel);
  arma::vec OutVector = arma::vec((CountVector.size()+2*KERNELSIZE), arma::fill::zeros);
  OutVector.subvec(0, KERNELSIZE-1) += CountVector.front();
  OutVector.subvec(endPoint, OutVector.size()-1) += CountVector.back();
  OutVector.subvec(KERNELSIZE,endPoint-1) += CountVector;
  Rcpp::NumericVector LamdaVec = Rcpp::wrap(arma::conv(OutVector, GaussKernel, "same")/(2*KERNELSIZE+1));
  Rcpp::NumericMatrix OutputMat(CountVector.size(), 2);
  for (unsigned int i = 0; i < CountVector.size(); ++i) {
    OutputMat(i, 0) = 0.5*R::qchisq(LowerBound, 2*LamdaVec[KERNELSIZE+i],1, 0);
    OutputMat(i, 1) = 0.5*R::qchisq(UpperBound, 2*LamdaVec[KERNELSIZE+i]+2,1, 0);
  }
  // add column names using upper and lower bounds
  std::stringstream strlow;
  strlow << "CI: " << alpha/2*100;
  std::string CIlow = strlow.str();
  std::stringstream strhigh;
  strhigh << "CI: " << (1-alpha/2)*100;
  std::string CIhigh = strhigh.str();
  Rcpp::colnames(OutputMat) = Rcpp::CharacterVector::create(CIlow, CIhigh);
  return OutputMat;
}

Rcpp::List SpikeCCFBatch(const Rcpp::NumericVector& Time,
                         const Rcpp::IntegerVector& UnitNr,
                         const double& WINDOW_LENGTH = 1,
                         const double& BIN_SIZE = 0.001,
                         const bool& BaselineFrequency = true,
                         const bool& ConfidenceInterval = true,
                         const double& ConfLevel = 0.95,
                         const double& SD = 0.6,
                         const double& CENTREMIN = 0.6,
                         const int& KERNELSIZE = 20) {
  Rcpp::IntegerVector UnitCollection = Rcpp::sort_unique(UnitNr);
  int MatLength = UnitCollection.size();
  MatLength = MatLength*MatLength;
  arma::mat CcfMatrix = arma::mat(roundf(WINDOW_LENGTH/BIN_SIZE)+1, MatLength, arma::fill::zeros);
  Rcpp::NumericMatrix CcfLower = Rcpp::NumericMatrix(roundf(WINDOW_LENGTH/BIN_SIZE)+1, MatLength);
  Rcpp::NumericMatrix CcfUpper = Rcpp::NumericMatrix(roundf(WINDOW_LENGTH/BIN_SIZE)+1, MatLength);  
  Rcpp::IntegerMatrix Units(MatLength, 2);
  arma::vec RandomBinCount(MatLength);
  Units(Rcpp::_,0) = Rcpp::rep_each(UnitCollection, UnitCollection.size());
  Units(Rcpp::_,1) = Rcpp::rep(UnitCollection, UnitCollection.size());
  for (int i = 0; i < MatLength; ++i) {
    Rcpp::NumericVector tmp1 = Time[UnitNr == Units(i,0)];
    Rcpp::NumericVector tmp2 = Time[UnitNr == Units(i,1)];
    double tmpMax;
    if(max(tmp1)<max(tmp2)) {
      tmpMax = max(tmp2);
    } else {
      tmpMax = max(tmp1);
    }
    double tmpMin;
    if(min(tmp1)<min(tmp2)) {
      tmpMin = min(tmp1);
    } else {
      tmpMin = min(tmp2);
    }
    CcfMatrix.unsafe_col(i) += Rcpp::as<arma::colvec>(spike_ccf(tmp1, tmp2, WINDOW_LENGTH, BIN_SIZE));
    if(ConfidenceInterval) {
      Rcpp::NumericMatrix CCFci = ConfIntPoisson(CcfMatrix.unsafe_col(i),
                                                 ConfLevel,
                                                 SD,
                                                 CENTREMIN,
                                                 KERNELSIZE);
      CcfLower(Rcpp::_,i) = CCFci(Rcpp::_,0);
      CcfUpper(Rcpp::_,i) = CCFci(Rcpp::_,1);
    }
    if(BaselineFrequency) {
      // compute random counts per bin
      if (Rcpp::is_false(Rcpp::all(tmp1==tmp2))) {
        RandomBinCount(i) = (tmp1.size()*tmp2.size())/(tmpMax-tmpMin)*BIN_SIZE;
      } else {
        RandomBinCount(i) = (tmp1.size()*tmp1.size()-tmp1.size())/(tmpMax-tmpMin)*BIN_SIZE;
      } 
    }
  }
  Rcpp::IntegerVector Xtmp = Rcpp::seq(-(roundf(WINDOW_LENGTH/BIN_SIZE)+1)/2,(roundf(WINDOW_LENGTH/BIN_SIZE)+1)/2);
  if(ConfidenceInterval & BaselineFrequency) {
    return Rcpp::List::create(Rcpp::Named("Units") = Units,
                              Rcpp::Named("CcfMatrix") = CcfMatrix,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE,
                              Rcpp::Named("LowerCI") = CcfLower,
                              Rcpp::Named("UpperCI") = CcfUpper,
                              Rcpp::Named("RandomBinCount") = RandomBinCount);
  } else if(!ConfidenceInterval & BaselineFrequency){
    return Rcpp::List::create(Rcpp::Named("Units") = Units,
                              Rcpp::Named("CcfMatrix") = CcfMatrix,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE,
                              Rcpp::Named("RandomBinCount") = RandomBinCount); 
  } else if(ConfidenceInterval & !BaselineFrequency){
    return Rcpp::List::create(Rcpp::Named("Units") = Units,
                              Rcpp::Named("CcfMatrix") = CcfMatrix,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE,
                              Rcpp::Named("LowerCI") = CcfLower,
                              Rcpp::Named("UpperCI") = CcfUpper); 
  } else if(!ConfidenceInterval & !BaselineFrequency){
    return Rcpp::List::create(Rcpp::Named("Units") = Units,
                              Rcpp::Named("CcfMatrix") = CcfMatrix,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE); 
  } else {
    Rcpp::stop("batch analysis failed");
  }
}


Rcpp::List SpikeCCFSingle(const Rcpp::NumericVector&  x,
                          const Rcpp::NumericVector&  y,
                          const double& WINDOW_LENGTH = 1,
                          const double& BIN_SIZE = 0.001,
                          bool BaselineFrequency = true,
                          bool ConfidenceInterval = true,
                          double ConfLevel = 0.95,
                          const double& SD = 0.6,
                          const double& CENTREMIN = 0.6,
                          const int& KERNELSIZE = 20) {
  double tmpMax;
  if(max(x)<max(y)) {
    tmpMax = max(y);
  } else {
    tmpMax = max(x);
  }
  double tmpMin;
  if(min(x)<min(y)) {
    tmpMin = min(x);
  } else {
    tmpMin = min(x);
  }
  
  Rcpp::IntegerVector HistOut = spike_ccf(x, y, WINDOW_LENGTH, BIN_SIZE);
  arma::vec tmp = Rcpp::as<arma::vec>(HistOut);
  int HistHalfSize = HistOut.size()/2;
  Rcpp::IntegerVector Xtmp = Rcpp::seq(-HistHalfSize,HistHalfSize);
  
  if(ConfidenceInterval and BaselineFrequency) {
    Rcpp::NumericMatrix OutputMat = ConfIntPoisson(tmp,
                                                   ConfLevel,
                                                   SD,
                                                   CENTREMIN,
                                                   KERNELSIZE);
    double RandomBins;
    if (Rcpp::is_false(Rcpp::all(x==y))) {
      RandomBins = (x.size()*y.size())/(tmpMax-tmpMin)*BIN_SIZE;
    } else {
      RandomBins = (x.size()*x.size()-x.size())/(tmpMax-tmpMin)*BIN_SIZE;
    } 
    return Rcpp::List::create(Rcpp::Named("CCF") = HistOut,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE,
                              Rcpp::Named("CI") = OutputMat,
                              Rcpp::Named("RandomBinCount") = RandomBins);  
  } if(BaselineFrequency) {
    double RandomBins;
    if (Rcpp::is_false(Rcpp::all(x==y))) {
      RandomBins = (x.size()*y.size())/(tmpMax-tmpMin)*BIN_SIZE;
    } else {
      RandomBins = (x.size()*x.size()-x.size())/(tmpMax-tmpMin)*BIN_SIZE;
    }
    return Rcpp::List::create(Rcpp::Named("CCF") = HistOut,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE,
                              Rcpp::Named("RandomBinCount") = RandomBins);
  } if(ConfidenceInterval) {
    Rcpp::NumericMatrix OutputMat = ConfIntPoisson(tmp,
                                                   ConfLevel,
                                                   SD,
                                                   CENTREMIN,
                                                   KERNELSIZE);
    return Rcpp::List::create(Rcpp::Named("CCF") = HistOut,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE,
                              Rcpp::Named("CI") = OutputMat);
  } else {
    return Rcpp::List::create(Rcpp::Named("CCF") = HistOut,
                              Rcpp::Named("xAxis") = Rcpp::as<Rcpp::NumericVector>(Xtmp)*BIN_SIZE);
  }
}


//' @title Spike cross-correlation function
//' 
//' The function will calculate the cross-correlation for spikes of the same or a different cell. 
//' The input arguments can be a vector containg timpoints (in seconds), two different vectors 
//' with timepoints or one vector with time points for all units and an integer vector containg 
//' the unit ID. The ouput will be depending on the input a single CCF or a list containing the 
//' CCF for all combinations in a matrix. Additional options include the computation of the confidence 
//' interval and the baseline activity.
//' 
//' @name SpikeCCF
//' @param x A numeric vector of times which has to be sorted in ascending order.
//' @param y A numeric vector of times which has to be sorted in ascending order.
//' @param UnitNr An integer vector containing the unit numbers in order of time occurence.
//' @param WINDOW_LENGTH An int as total window length in seconds (default = 1).
//' @param BIN_SIZE A double indicating the size of bins in seconds (default = 0.001).
//' @param BaselineFrequency A bool to indicate whether base line activity should be estimated (default = TRUE).
//' @param ConfidenceInterval A bool to indicate whether confidence interval should be estimated (default = TRUE).
//' @param ConfLevel A double indicating the confidence-level (default = 0.95).
//' @param SD A double as standard deviation for a gaussian shape parameter (default = 0.6).
//' @param CENTREMIN A double as shape parameter determining the strength of centre exclusion (default = 0.6).
//' @param KERNELSIZE A double as length parameter for gaussian kernel (2*KERNELSIZE+1, default = 20).
//' 
//' @return Returns a list containing counts per bin, axis, random bin count, confidence-intervals with counts per bin.
//' @export
// [[Rcpp::export]]
Rcpp::List SpikeCCF(const Rcpp::NumericVector  x,
                    Rcpp::Nullable<Rcpp::NumericVector> y = R_NilValue,
                    Rcpp::Nullable<Rcpp::IntegerVector> UnitNr = R_NilValue,
                    const double WINDOW_LENGTH = 1,
                    const double BIN_SIZE = 0.001,
                    const bool BaselineFrequency = true,
                    const bool ConfidenceInterval = true,
                    const double ConfLevel = 0.95,
                    const double SD = 0.6,
                    const double CENTREMIN = 0.6,
                    const int KERNELSIZE = 20) {
  if(y.isNotNull() & !UnitNr.isNotNull()) {
    if(Rcpp::as<Rcpp::NumericVector>(y).size()>1) {
      return SpikeCCFSingle(x,
                            Rcpp::as<Rcpp::NumericVector>(y),
                            WINDOW_LENGTH,
                            BIN_SIZE,
                            BaselineFrequency,
                            ConfidenceInterval,
                            ConfLevel,
                            SD,
                            CENTREMIN,
                            KERNELSIZE);
    } else {
      Rcpp::stop("y does not contain engough values");
    }
  } else if(UnitNr.isNotNull() & !y.isNotNull()) {
    if(Rcpp::as<Rcpp::IntegerVector>(UnitNr).size()>1) {
      return SpikeCCFBatch(x,
                           Rcpp::as<Rcpp::IntegerVector>(UnitNr),
                           WINDOW_LENGTH,
                           BIN_SIZE,
                           BaselineFrequency,
                           ConfidenceInterval,
                           ConfLevel,
                           SD,
                           CENTREMIN,
                           KERNELSIZE);
    } else {
      Rcpp::stop("UnitNr does not contain engough values");
    }
  } else if(!UnitNr.isNotNull() & !y.isNotNull()) {
    if(x.size()>1) {
      return SpikeCCFSingle(x,
                            x,
                            WINDOW_LENGTH,
                            BIN_SIZE,
                            BaselineFrequency,
                            ConfidenceInterval,
                            ConfLevel,
                            SD,
                            CENTREMIN,
                            KERNELSIZE);
    } else {
      Rcpp::stop("Missing arguments"); 
    }
  } else if(UnitNr.isNotNull() & y.isNotNull()) {
    Rcpp::stop("Too many arguments: 'y' for single CCF and 'UnitNr' for batch analysis");
  } else {
    Rcpp::stop("Check inputs");
  }
}