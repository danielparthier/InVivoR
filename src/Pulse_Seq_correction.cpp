// Function to detect stimulus times, frequencies, types, and blocks using the raw stimulus input and a filtered version.
// Output of function is a list of two matrices including the single pulse information and the block information.

#include <RcppArmadillo.h>
#include <omp.h>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]

//' Butterworth filter c++
//' 
//' This function returns a filtered Signal.
//'
//' @param InputSignal A complex matrix from FFTW.
//' @param SamplingFrequency A double indicating sampling frequency.
//' @param ORDER An int as filtering order (default = 2).
//' @param f0 A double as cutoff frequency (default = 10).
//' @param type A string indicating the filter type ("low", "high"). The default is "low".
//' @param CORES An int indicating the number of threads used (default = 1).
//' @return Filtered signal as numeric vector.
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector BWFiltCpp(arma::vec& InputSignal,
                              const double& SamplingFrequency,
                              const int& ORDER = 2,
                              const double& f0 = 10,
                              const std::string type = "low",
                              const int& CORES = 1) {
  omp_set_num_threads(CORES);
  int InputLength = std::pow(2, std::ceil(std::log2(InputSignal.size())));
  arma::vec InputSignalPadded = arma::zeros<arma::vec>(InputLength);
  int signalLength = InputSignal.size();
  int startSignal = (InputLength-signalLength)*0.5;
  InputSignalPadded.subvec(startSignal, size(InputSignal)) = InputSignal;
  arma::cx_mat InputFFT = arma::fft(InputSignalPadded);
  int OrderTerm = ORDER*2;
  int SignalLength = InputFFT.size();
  if (f0 > 0) {
    int HalfLength = SignalLength*0.5;
    double BinWidth = SamplingFrequency/SignalLength;
    if(type.compare("low") == 0) {
      BinWidth = BinWidth/f0;
#pragma omp parallel for shared(f0, HalfLength, InputFFT, BinWidth, OrderTerm, SignalLength) schedule(dynamic) default(none)
      for (int i=0; i<HalfLength; ++i) {
        double gain = 1/std::sqrt(1+std::pow((BinWidth * i), OrderTerm));
        InputFFT.at(i) *= gain;
        InputFFT.at(SignalLength-(i)) *= gain;
      } 
    } if(type.compare("high") == 0) {
      BinWidth = f0/BinWidth;
#pragma omp parallel for shared(f0, HalfLength, InputFFT, BinWidth, OrderTerm, SignalLength) schedule(dynamic) default(none)
      for (int i=0; i<HalfLength; ++i) {
        double gain = 1/std::sqrt(1+std::pow(f0/(BinWidth * (i+1)), OrderTerm));
        InputFFT.at(i) *= gain;
        InputFFT.at(SignalLength-(i)) *= gain;
      }
    } if((type.compare("low") != 0) and (type.compare("high") != 0)) {
      Rcpp::stop("No valid type parameter.");
    }
  }
  InputSignalPadded = arma::real(arma::ifft(InputFFT));
  InputSignalPadded = InputSignalPadded.subvec(startSignal, size(InputSignal));
  return Rcpp::NumericVector(InputSignalPadded.begin(),InputSignalPadded.end());
}

// housekeeping functions
// function for median calculation (faster than R implementation)
double cpp_med(Rcpp::NumericVector xx) {
  Rcpp::NumericVector x = Rcpp::clone(xx);
  std::size_t n = x.size() / 2;
  std::nth_element(x.begin(), x.begin() + n, x.end());
  
  if (x.size() % 2) return x[n]; 
  return (x[n] + *std::max_element(x.begin(), x.begin() + n)) / 2.;
}

//' Stimulus sequence
//' 
//' This function analyses stimulus time series and extracts features which 
//' are returned in a list. Part of the output is the single stimulus feature
//' and the second part is a stimulus block summary.
//'
//' @param raw A numeric vector which is the stimulation over time as continues series.
//' @param sampling_frequency An integer indicating the sampling frequency.
//' @param threshold A double indicating the threshold of stimulus detection.
//' @param max_time_gap A double indicating the maximum time between blocks.
//' @return Returns a list with a matrix showing single pulse properperties and a matrix with block properties.
//' @export
// [[Rcpp::export]]
Rcpp::List StimulusSequence(Rcpp::NumericVector& raw,
                            int& sampling_frequency,
                            double& threshold,
                            const double& max_time_gap) {
  arma::vec rawArma = Rcpp::as<arma::vec>(raw);
  Rcpp::NumericVector filt = BWFiltCpp(rawArma, sampling_frequency, 2, 1000, "low", 1);
  // thresholding of raw, filtered stimulus trace & finding primary onsets and ends of stimuli
  int CorrectionCount = 1;
  new_try:
  int end_p = raw.size();
  std::vector<int> out_vec_onset_raw;
  std::vector<int> out_vec_end_raw;
  std::vector<int> out_vec_onset_filt;
  std::vector<int> out_vec_end_filt;
  for(R_xlen_t i=0; i < end_p; ++i) {
    if(((raw[i] > threshold)==true) & (i > 0)){
      if((raw[i-1] < threshold) & (raw[i+1] > threshold)) {
        out_vec_onset_raw.push_back(i);
      } else if((raw[i-1] > threshold) & (raw[i+1] < threshold)){
        out_vec_end_raw.push_back(i);
      }
    }
    if(((filt[i] > threshold)==true) & (i > 0)){
      if((filt[i-1] < threshold) & (filt[i+1] > threshold)) {
        out_vec_onset_filt.push_back(i);
      } else if((filt[i-1] > threshold) & (filt[i+1] < threshold)){
        out_vec_end_filt.push_back(i);
      }
    }
  }
  Rcpp::IntegerVector stim_pulse_start_raw = Rcpp::wrap(out_vec_onset_raw);
  Rcpp::IntegerVector stim_pulse_start = Rcpp::wrap(out_vec_onset_filt);
  Rcpp::IntegerVector stim_pulse_end_raw = Rcpp::wrap(out_vec_end_raw);
  Rcpp::IntegerVector stim_pulse_end = Rcpp::wrap(out_vec_end_filt);
  
  int l_stim_pulse_start(stim_pulse_start.size());
  Rcpp::NumericMatrix stim_mat(l_stim_pulse_start, 18); // initialising output matrix
  // Matrix: rows: left, right; cols: including 0 (zero_threshold), 25, 50, 75, 95 (Peak)
  Rcpp::NumericMatrix shape_mat(3, 5);
  shape_mat(0,0) = 0;
  shape_mat(0,1) = 0.25;
  shape_mat(0,2) = 0.50;
  shape_mat(0,3) = 0.75;
  shape_mat(0,4) = 0.95;
  Rcpp::NumericVector diff_point_collect_y = Rcpp::diff(shape_mat.row(0));
  
  int corr_start = 0, corr_end = 0;
  int peak_loc = 0;
  double zero_threshold = cpp_med(filt);
  double peak_amp = 0.0;
  bool pulse = false;
  bool sine = false;
  bool found0 = false, found25 = false, found50 = false, found75 = false, found95 = false;
  bool front_pulse = false, back_pulse = false;
  bool plateau_phase = false;
  bool ramp_front = false, ramp_end = false;
  
  // loop through "pulses" and detect type of pulse and start- and endpoint
  for (R_xlen_t i = 0; i < l_stim_pulse_start; ++i) {
    corr_start = stim_pulse_start_raw[which_min(abs(stim_pulse_start_raw-stim_pulse_start[i]))];
    corr_end = stim_pulse_end_raw[which_min(abs(stim_pulse_end_raw-stim_pulse_end[i]))];
    if(corr_end<=corr_start) {
      Rcpp::warning("Start and end of detection cause conflict! ... Check raw input and adjust threshold if necessary");
      if(threshold*1.5<Rcpp::max(raw)) {
        threshold=threshold*1.5;
        std::string WarningString = "New Threshold = " + std::to_string(threshold) + " at iteration: " + std::to_string(CorrectionCount) + "/10";
        Rcpp::warning(WarningString);
        if(CorrectionCount == 10) {
          Rcpp::stop("Threshold adjustment failed");
        }
        ++CorrectionCount;
      } else {
        Rcpp::stop("Threshold adjustment failed");
      }
      goto new_try;
    }
    int peak_loc_tmp = Rcpp::which_max(filt[Rcpp::seq(corr_start, corr_end)])+corr_start;
    double tmp_max = Rcpp::max(raw[Rcpp::seq(corr_start, corr_end)]);
    double tmp_25 = (tmp_max-zero_threshold)*shape_mat(0,1)+zero_threshold, tmp_50 = (tmp_max-zero_threshold)*shape_mat(0,2)+zero_threshold, tmp_75 = (tmp_max-zero_threshold)*shape_mat(0,3)+zero_threshold, tmp_95 = (tmp_max-zero_threshold)*shape_mat(0,4)+zero_threshold;
    // generate shape matrix to identify type of input
    for(R_xlen_t forward_run = peak_loc_tmp; forward_run < peak_loc_tmp+sampling_frequency*2; ++forward_run) {
      double forward_run_tmp = raw[forward_run];
      if((forward_run_tmp<tmp_95) & (found95 == false)){
        shape_mat(2,4) = forward_run;
        found95 = true;
        --forward_run;
      } else if((forward_run_tmp<tmp_75) & (found75 == false)){
        shape_mat(2,3) = forward_run;
        found75 = true;
        --forward_run;
      } else if((forward_run_tmp<tmp_50) & (found50 == false)){
        shape_mat(2,2) = forward_run;
        found50 = true;
        --forward_run;
      } else if((forward_run_tmp<tmp_25) & (found25 == false)){
        shape_mat(2,1) = forward_run;
        found25 = true;
        --forward_run;
      } else if((forward_run_tmp<zero_threshold) & (found0 == false)){
        shape_mat(2,0) = forward_run;
        found0 = true;
        --forward_run;
      } else if(found0 & found25 & found50 & found75 & found95 & true){
        forward_run = peak_loc_tmp+sampling_frequency*2;
      }
    }
    
    // reset for backwards run
    found0 = false, found25 = false, found50 = false, found75 = false, found95 = false;
    for(R_xlen_t back_run = peak_loc_tmp; back_run > peak_loc_tmp-sampling_frequency*2; --back_run) {
      double back_run_tmp = raw[back_run];
      if((back_run_tmp<tmp_95) & (found95 == false) ){
        shape_mat(1,4) = back_run;
        found95 = true;
        ++back_run;
      } else if((back_run_tmp<tmp_75) & (found75 == false)){
        shape_mat(1,3) = back_run;
        found75 = true;
        ++back_run;
      } else if((back_run_tmp<tmp_50) & (found50 == false)){
        shape_mat(1,2) = back_run;
        found50 = true;
        ++back_run;
      } else if((back_run_tmp<tmp_25) & (found25 == false)){
        shape_mat(1,1) = back_run+1;
        found25 = true;
        ++back_run;
      } else if((back_run_tmp<zero_threshold) & (found0 == false)){
        shape_mat(1,0) = back_run;
        found0 = true;
        ++back_run;
      } else if(found0 & found25 & found50 & found75 & found95 & true){
        back_run = peak_loc_tmp-sampling_frequency*2;
      }
    }  
    
    found0 = false, found25 = false, found50 = false, found75 = false, found95 = false;
    
    // compute time point differences and slope to find pulse (fast changes)
    Rcpp::NumericVector diff_forward = Rcpp::diff(shape_mat.row(2));
    Rcpp::NumericVector diff_backward = Rcpp::diff(shape_mat.row(1));
    Rcpp::NumericVector slope_forward = diff_point_collect_y/diff_forward;
    Rcpp::NumericVector slope_backward = diff_point_collect_y/diff_backward;
    corr_start = shape_mat(1,0);
    corr_end = shape_mat(2,0);
    
    if(is_true(any(diff_backward == 0))){
      front_pulse = true;
    } else {
      front_pulse = false;
    }
    if(is_true(any(diff_forward == 0))){
      back_pulse = true;
    } else {
      back_pulse = false;
    }
    
    double slope_forward_rmse = ::sqrt(sum((slope_forward-mean(slope_forward))*(slope_forward-mean(slope_forward)))/4);
    double slope_backward_rmse = ::sqrt(sum((slope_backward-mean(slope_backward))*(slope_backward-mean(slope_backward)))/4);
    if((front_pulse & back_pulse) == true) {
      // pulse detection
      corr_start = shape_mat(1,1);
      corr_end = shape_mat(2,1);
      peak_loc = corr_start;
      peak_amp = cpp_med(raw[Rcpp::seq(corr_start, corr_end)]);
      pulse =true;
      sine = false;
      ramp_front = false;
      ramp_end = false;
      plateau_phase = true;
      //ramp detection is experimental and needs further testing (slope rmse approach)
    } else if(((front_pulse & back_pulse) == false) & (slope_forward_rmse > 0.0001) & (slope_backward_rmse > 0.0001)) {
      for(R_xlen_t back_run = shape_mat(1,1); back_run > shape_mat(1,1)-sampling_frequency*2; --back_run) {
        if(filt[back_run]-(filt[back_run-1])<0){
          corr_start = back_run;
          back_run = shape_mat(1,1)-sampling_frequency*2;
        }
      }
      for(R_xlen_t forward_run = shape_mat(2,1); forward_run < shape_mat(2,1)+sampling_frequency*2; ++forward_run) {
        if(filt[forward_run]-(filt[forward_run-1])>0){
          corr_end = forward_run;
          forward_run = shape_mat(2,1)+sampling_frequency*2;
        }
      }
      peak_loc = peak_loc_tmp;
      peak_amp = Rcpp::max(filt[Rcpp::seq(corr_start, corr_end)]);
      pulse = false;
      sine = true;
      ramp_front = false;
      ramp_end = false;
      plateau_phase = false;
    } else if((back_pulse == true) & (front_pulse == false) & (slope_backward_rmse < 0.0001)) {
      int peak_loc_leap = round((shape_mat(1,4)-shape_mat(1,0))/shape_mat(0,4));
      // use 10% range to find peak amp
      if(peak_loc_leap*110+corr_start > corr_end) {
        peak_loc = peak_loc_tmp;
        peak_amp = Rcpp::max(raw[Rcpp::seq(corr_start, corr_end)]);
        plateau_phase = false;
      } else {
        peak_loc = peak_loc_leap*100+corr_start;
        peak_amp = raw[peak_loc];
        plateau_phase = true;
      }
      pulse = false;
      sine = false;
      ramp_front = true;
      ramp_end = false;
    } else if((back_pulse == false) & (front_pulse == true) & (slope_forward_rmse < 0.0001)) {
      int peak_loc_leap = round((shape_mat(1,4)-shape_mat(1,0))/shape_mat(0,4));
      // use 10% range to find peak amp
      if(peak_loc_leap*110-corr_end < corr_start) {
        peak_loc = peak_loc_tmp;
        peak_amp = Rcpp::max(raw[Rcpp::seq(corr_start, corr_end)]);
        plateau_phase = false;
      } else {
        peak_loc = peak_loc_leap*100-corr_end;
        peak_amp = raw[peak_loc];
        plateau_phase = true;
      }
      pulse = false;
      sine = false;
      ramp_front = true;
      ramp_end = false;
    }
    // fill output matrix with parameters and adjust for indices in R
    stim_mat(i,0) = corr_start+1;
    stim_mat(i,1) = corr_end+1;
    stim_mat(i,2) = peak_loc+1;
    stim_mat(i,3) = peak_amp;
    stim_mat(i,4) = pulse;
    stim_mat(i,5) = sine;
    stim_mat(i,6) = corr_end-corr_start;
    stim_mat(i,7) = ramp_front;
    stim_mat(i,8) = ramp_end;
    stim_mat(i,9) = plateau_phase;
  }
  int block_number = 1;
  int pulse_number_block_intern = 0;
  int hyper_block_number = 1;
  double freq_out = 0;
  double current_hyper_freq = 0;
  double start_stim_freq_post = round(sampling_frequency/(stim_mat(1,2)-stim_mat(0,2))*10)/10;
  for(R_xlen_t i = 0; i < l_stim_pulse_start; ++i) {
    stim_mat(i, 13) = i+1;
    if((i > 1) & (i < l_stim_pulse_start-2)) {
      double stim_freq_post = sampling_frequency/(stim_mat(i+1,2)-stim_mat(i,2));
      double stim_freq_pre = sampling_frequency/(stim_mat(i,2)-stim_mat(i-1,2));
      double stim_freq_pre2 = sampling_frequency/(stim_mat(i-1,2)-stim_mat(i-2,2));
      // compare local frequencies and use 5% detection error as tolerance value
      if(fabs((stim_freq_pre-stim_freq_pre2)/(stim_freq_pre+stim_freq_pre2)) < 0.05 and fabs((stim_freq_pre-stim_freq_post)/(stim_freq_pre+stim_freq_post)) < 0.05) {
        //pre2==pre==post
        //round to one digit
        if(stim_freq_post > 10) {
          freq_out = round(stim_freq_post);
        } else if(stim_freq_post > 0.5) {
          freq_out = round(stim_freq_post*10)/10;
        } else {
          freq_out = round(stim_freq_post*100)/100;
        }
      } else if(fabs((stim_freq_pre2-stim_freq_pre)/(stim_freq_pre2+stim_freq_pre)) < 0.05 and fabs((stim_freq_pre-stim_freq_post)/(stim_freq_pre+stim_freq_post)) > 0.05) {
        //pre2==pre!=post
        if(stim_freq_pre > stim_freq_post) {
          if(stim_freq_pre > 10) {
            freq_out = round(stim_freq_pre);
          } else if(stim_freq_pre > 0.5) {
            freq_out = round(stim_freq_pre*10)/10;
          } else {
            freq_out = round(stim_freq_pre*100)/100;
          }
        } else {
          if(stim_freq_post > 10) {
            freq_out = round(stim_freq_post);
          } else if(stim_freq_post > 0.5) {
            freq_out = round(stim_freq_post*10)/10;
          } else {
            freq_out = round(stim_freq_post*100)/100;
          }
          ++block_number;
          pulse_number_block_intern = 0;
        }
      } else if(fabs((stim_freq_pre2-stim_freq_pre)/(stim_freq_pre2+stim_freq_pre)) > 0.05 and fabs((stim_freq_pre-stim_freq_post)/(stim_freq_pre+stim_freq_post)) < 0.05) {
        //pre2!=pre==post
        if(stim_freq_post > 10) {
          freq_out = round(stim_freq_post);
        } else if(stim_freq_post > 0.5) {
          freq_out = round(stim_freq_post*10)/10;
        } else {
          freq_out = round(stim_freq_post*100)/100;
        }
         if((stim_mat(i,4)!= stim_mat(i-1,4)) | (stim_mat(i,5)!= stim_mat(i-1,5)) | (stim_mat(i,6)*0.05 < abs(stim_mat(i,6)-stim_mat(i-1,6))) |(stim_mat(i,9)!= stim_mat(i-1,9)) | (round(stim_mat(i,3)*10) != round(stim_mat(i-1,3)*10)) | (freq_out != stim_mat(i-1,10))) {
           ++block_number;
           pulse_number_block_intern = 0;
         }
      } else if(fabs((stim_freq_pre2-stim_freq_post)/(stim_freq_pre2+stim_freq_post)) < 0.05 and fabs((stim_freq_pre-stim_freq_post)/(stim_freq_pre+stim_freq_post)) > 0.05) {
        //pre2!=pre!=post pre2==post
        if(stim_freq_pre2 > 10) {
          freq_out = round(stim_freq_pre2);
        } else if(stim_freq_post > 0.5) {
          freq_out = round(stim_freq_pre2*10)/10;
        } else {
          freq_out = round(stim_freq_pre2*100)/100;
        }
        ++block_number;
        pulse_number_block_intern = 0;
      } else if(fabs((stim_freq_pre2-stim_freq_post)/(stim_freq_pre2+stim_freq_post)) > 0.05 and fabs((stim_freq_pre-stim_freq_post)/(stim_freq_pre+stim_freq_post)) > 0.05 and fabs((stim_freq_pre2-stim_freq_pre)/(stim_freq_pre2+stim_freq_pre)) > 0.05) {
        //pre2!=pre!=post pre2==post
        if(stim_freq_post > 10) {
          freq_out = round(stim_freq_post);
        } else if(stim_freq_post > 0.5) {
          freq_out = round(stim_freq_post*10)/10;
        } else {
          freq_out = round(stim_freq_post*100)/100;
        }
        if((stim_mat(i,4)!= stim_mat(i-1,4)) | (stim_mat(i,5)!= stim_mat(i-1,5)) | (stim_mat(i,6)*0.05 < abs(stim_mat(i,6)-stim_mat(i-1,6))) |(stim_mat(i,9)!= stim_mat(i-1,9)) | (round(stim_mat(i,3)*10) != round(stim_mat(i-1,3)*10)) | (freq_out != stim_mat(i-1,10))) {
          ++block_number;
          pulse_number_block_intern = 0;
        }
      } else {
        ++block_number;
        pulse_number_block_intern = 0;
      }
    } else if(i < 2) {
      if(i == 0) {
        if(start_stim_freq_post > 10) {
          freq_out = round(start_stim_freq_post);
        } else if(start_stim_freq_post > 0.5) {
          freq_out = round(start_stim_freq_post*10)/10;
        } else {
          freq_out = round(start_stim_freq_post*100)/100;
        }
      } else {
        double stim_freq_post = sampling_frequency/(stim_mat(i+1,2)-stim_mat(i,2));
        if(fabs(stim_freq_post-start_stim_freq_post) <  (stim_freq_post+start_stim_freq_post)/0.025) {
          if((stim_freq_post+start_stim_freq_post)/2 > 10) {
            freq_out = round((stim_freq_post+start_stim_freq_post)/2);
          } else if((stim_freq_post+start_stim_freq_post)/2 > 0.5) {
            freq_out = round((stim_freq_post+start_stim_freq_post)*5)/10;
          } else {
            freq_out = round((stim_freq_post+start_stim_freq_post)*50)/100;
          }
          if((stim_mat(i,4)!= stim_mat(i-1,4)) | (stim_mat(i,5)!= stim_mat(i-1,5)) | (stim_mat(i,6)*0.05 < abs(stim_mat(i,6)-stim_mat(i-1,6))) |(stim_mat(i,9)!= stim_mat(i-1,9)) | (round(stim_mat(i,3)*10) != round(stim_mat(i-1,3)*10)) | (freq_out != stim_mat(i-1,10))) {
            ++block_number;
            pulse_number_block_intern = 0;
          }        
        }
      }
    } else if(i > l_stim_pulse_start-3) {
      double stim_freq_pre2 = sampling_frequency/(stim_mat(i-1,2)-stim_mat(i-2,2));
      double stim_freq_pre = sampling_frequency/(stim_mat(i,2)-stim_mat(i-1,2));
      if(fabs(stim_freq_pre-stim_freq_pre2) < stim_freq_pre2/0.025) {
        if(stim_freq_pre > 10) {
          freq_out = round(stim_freq_pre);
        } else if(stim_freq_pre > 0.5) {
          freq_out = round(stim_freq_pre*10)/10;
        } else {
          freq_out = round(stim_freq_pre*100)/100;
        }
        if((stim_mat(i,4)!= stim_mat(i-1,4)) | (stim_mat(i,5)!= stim_mat(i-1,5)) | (stim_mat(i,6)*0.05 < abs(stim_mat(i,6)-stim_mat(i-1,6))) |(stim_mat(i,9)!= stim_mat(i-1,9)) | (round(stim_mat(i,3)*10) != round(stim_mat(i-1,3)*10)) | (freq_out != stim_mat(i-1,10))) {
          ++block_number;
          pulse_number_block_intern = 0;
        }
      }
    }
    stim_mat(i, 10) = freq_out;
    stim_mat(i, 11) = block_number;
    stim_mat(i, 12) = ++pulse_number_block_intern;
    stim_mat(i-1, 14) = (stim_mat(i-1, 12)==stim_mat(i, 12));
    if(stim_mat(i, 12) == 2) {
      stim_mat(i-1, 10) = freq_out;
    }
    if(i > l_stim_pulse_start-3){
      stim_mat(i-1, 14) = (stim_mat(i-1, 12)==stim_mat(i, 12));
      if(i == l_stim_pulse_start-1 and stim_mat(i, 12)==1) {
        stim_mat(i, 14) = 1;
      }
    }
  }
  
  int next_i = 0;
  int next_i_nd = 0;
  current_hyper_freq = -1;
  double first_hyper_freq = -1;
  double second_hyper_freq = -1;
  double future_hyper_freq = -1;
  for(R_xlen_t i = l_stim_pulse_start-1; i > -1; --i) {
    // enter detection 
    if(stim_mat(i, 12) == 1 and i > 0 and stim_mat(i, 14) == 0) {
      next_i = i-1;
      while(stim_mat(next_i, 12) != 1 and next_i > 0) {
        //search for next block
        --next_i;
        if(stim_mat(next_i, 12) == 1) {
          //compute frequency between 2 blocks
          first_hyper_freq = 1/((stim_mat(i, 1)-stim_mat(next_i, 1))/sampling_frequency);
          if(first_hyper_freq > 10) {
            first_hyper_freq = round(first_hyper_freq);
          } else if(first_hyper_freq > 0.5) {
            first_hyper_freq = round(first_hyper_freq*10)/10;
          } else {
            first_hyper_freq = round(first_hyper_freq*100)/100;
          }
          next_i_nd = next_i-1;
          while(stim_mat(next_i_nd, 12) != 1 and next_i_nd > 0) {
            //search for next block to verify change
            --next_i_nd;
            if(stim_mat(next_i_nd, 12) == 1) {
              second_hyper_freq = 1/((stim_mat(next_i, 1)-stim_mat(next_i_nd, 1))/sampling_frequency);
              if(second_hyper_freq > 10) {
                second_hyper_freq = round(second_hyper_freq);
              } else if(second_hyper_freq > 0.5) {
                second_hyper_freq = round(second_hyper_freq*10)/10;
              } else {
                second_hyper_freq = round(second_hyper_freq*100)/100;
              }
            }
          }
          int future_i = next_i+1;
          while(stim_mat(future_i, 12) != 1 and future_i < l_stim_pulse_start) {
            ++future_i;
          }
          future_hyper_freq = 1/((stim_mat(future_i, 1)-stim_mat(next_i, 1))/sampling_frequency);
          if(future_hyper_freq > 10) {
            future_hyper_freq = round(future_hyper_freq);
          } else if(future_hyper_freq > 0.5) {
            future_hyper_freq = round(future_hyper_freq*10)/10;
          } else {
            future_hyper_freq = round(future_hyper_freq*100)/100;
          }
        }
      }
      if(first_hyper_freq < future_hyper_freq) {
        current_hyper_freq = future_hyper_freq;
      } else if(first_hyper_freq==second_hyper_freq) {
        current_hyper_freq = first_hyper_freq;
      } else if(first_hyper_freq == future_hyper_freq) {
        current_hyper_freq = first_hyper_freq;
      }
      if(stim_mat(i, 16) == stim_mat(i+1, 16)){
        int recover_i = i;
        
        //backtrack frequencies to avoid gaps or overlap
        while(stim_mat(recover_i, 16) == stim_mat(recover_i+1, 16)) {
          ++recover_i;
          stim_mat(recover_i, 15) = stim_mat(i, 15);
        }
      }
      if(first_hyper_freq<0.5 or (first_hyper_freq != second_hyper_freq and first_hyper_freq != future_hyper_freq and stim_mat(i-1, 14) == 0)) {
        if(stim_mat(i+1, 16) == hyper_block_number) {
          current_hyper_freq = stim_mat(i+1, 15);
        }
        stim_mat(i, 15) = current_hyper_freq;
        stim_mat(i, 16) = hyper_block_number;
        --hyper_block_number;
      }
    } else if (stim_mat(i, 14) == 1) {
      stim_mat(i, 15) = stim_mat(i, 10);
      stim_mat(i, 16) = stim_mat(i+1, 16)-1;
      --hyper_block_number;
    } else {
      
      stim_mat(i, 15) = current_hyper_freq;
      stim_mat(i, 16) = hyper_block_number;
    }
    if(stim_mat(i, 12)==2){
      stim_mat(i-1, 15) = current_hyper_freq;
      stim_mat(i-1, 16) = hyper_block_number;
    }
  }

  //fix end frequency
  if(stim_mat(l_stim_pulse_start-1, 15) == -1) {
    int i = l_stim_pulse_start-1;
    while((stim_mat(i, 15) == -1)){
      stim_mat(i, 15) = stim_mat(i, 10);
      if(i > 0) {
        --i;
      } else {
        break;
      }
    }
    int pulse_count = 1;
    for(R_xlen_t fix_i = i; fix_i < l_stim_pulse_start; ++fix_i){
      stim_mat(fix_i, 17) = pulse_count;
      ++pulse_count;
    }
  }
  
  int block_count = Rcpp::min(stim_mat(Rcpp::_,16))-1;
  for(R_xlen_t i = 0; i < l_stim_pulse_start; ++i) {
    stim_mat(i, 16) = stim_mat(i, 16)-block_count;
  }
  
  /// loop over hyper blocks --> use as index and vec[hyperblock] check for ones in block --> if more than one stick with freq otherwise use single freq as hyper
  int max_hyper_block = Rcpp::max(stim_mat(Rcpp::_, 16));
  int block_id = 0;
  for(R_xlen_t hyper_block_i = 1; hyper_block_i < max_hyper_block+1; ++hyper_block_i){
    int old_start = block_id;
    int inside_block_count = 0;
    int hyper_pulse_nr = 1;
    while(stim_mat(block_id, 16)==hyper_block_i) {
      if(stim_mat(block_id, 12) == 1) {
        ++inside_block_count;
      }
      stim_mat(block_id, 17) = hyper_pulse_nr;
      ++hyper_pulse_nr;
      ++block_id;
    }
    if(inside_block_count == 1) {
      for(R_xlen_t fill_mat = old_start; fill_mat < block_id; ++fill_mat) {
        stim_mat(fill_mat, 15) = stim_mat(fill_mat, 10);
      }
    }
  }
  colnames(stim_mat) = Rcpp::CharacterVector::create("onset",
           "end",
           "peak_loc",
           "peak_amp",
           "pulse",
           "sine",
           "stim_length",
           "ramp_front",
           "ramp_end",
           "plateau",
           "frequency",
           "block_nr",
           "pulse_nr_block",
           "pulse_nr",
           "pulse_isolation",
           "hyper_block_freq",
           "hyper_block_nr",
           "hyper_block_pulse_nr");
  
  // compute frequency and block number
  int max_block = Rcpp::max(stim_mat(Rcpp::_, 16));
  Rcpp::NumericMatrix block_pos(max_block, 17);
  R_xlen_t stim_mat_end = stim_mat.nrow();
  
  int block_start = 0;
  int block_end = 0;
  int end_pulse = 0;
  //adding the block edges for every stimulation block
  for(R_xlen_t block_i = 0; block_i < stim_mat_end; ++block_i) {
    if(stim_mat(block_i, 17) == 1) {
      //block start
      block_pos(block_start, 0) = round(stim_mat(block_i, 2)-0.5*sampling_frequency/stim_mat(block_i, 15));
      //start pulse
      block_pos(block_start, 2) = block_i+1;
      //isolated block
      block_pos(block_start, 8) = stim_mat(block_i, 14);
      //frequency
      block_pos(block_start, 9) = stim_mat(block_i, 10);
      //sine wave
      block_pos(block_start, 10) = stim_mat(block_i,5);
      //pulse
      block_pos(block_start, 11) = stim_mat(block_i,4);
      // ramp_front
      block_pos(block_start, 12) = stim_mat(block_i,7);
      // ramp_end
      block_pos(block_start, 13) = stim_mat(block_i,8);
      //hyper block number
      block_pos(block_start, 14) = stim_mat(block_i, 16);
      //hyper block freq
      block_pos(block_start, 15) = stim_mat(block_i, 15);
      //stimulation length of pulse or sine etc.
      block_pos(block_start, 16) = round(stim_mat(block_i, 6)/sampling_frequency*1000)/1000;
      int while_block_i = block_i+1;
      end_pulse = block_i;
      if(stim_mat(block_i, 10) != stim_mat(block_i, 15)) {
        //burst stimulation
        while(while_block_i < stim_mat_end and stim_mat(block_i, 16) == stim_mat(while_block_i, 16) and stim_mat(block_i, 14) != 1){
          if(stim_mat(while_block_i, 12) == 1){
            end_pulse = while_block_i;
          }
          ++while_block_i; 
        }
        //burst stim
        block_pos(block_start, 7) = 1;
      } else {
        //regular stimulation
        while(while_block_i < stim_mat_end and stim_mat(block_i, 16) == stim_mat(while_block_i, 16) and stim_mat(block_i, 14) != 1){
          end_pulse = while_block_i;
          ++while_block_i; 
        }
      }
      //block end
      block_pos(block_end,1) = round(stim_mat(end_pulse, 2)+0.5*sampling_frequency/stim_mat(end_pulse, 15));
      //end pulse
      block_pos(block_end,3) = end_pulse+1;
      ++block_end;
      ++block_start;
      block_i = end_pulse;
    }
  }

  for(R_xlen_t i = 0; i < block_pos.nrow(); ++i) {
    //finding block length
    //center
    block_pos(i, 4) = block_pos(i, 0)+(block_pos(i, 1)-block_pos(i, 0))/2;
    if(block_pos(i, 14) >= 1/(max_time_gap*2)) {
      //left edge    
      block_pos(i, 5) = block_pos(i, 0)-sampling_frequency*max_time_gap/2;
      //right edge
      block_pos(i, 6) = block_pos(i, 1)+sampling_frequency*max_time_gap/2;
    } else if(block_pos(i, 14)<1/(max_time_gap*2)) {
      //left edge
      block_pos(i, 5) = stim_mat(block_pos(i, 2)-1,2)-stim_mat(block_pos(i, 2)-1,6)/2-sampling_frequency*max_time_gap/2;
      //right edge
      block_pos(i, 6) = stim_mat(block_pos(i, 3)-1,2)-stim_mat(block_pos(i, 3)-1,6)/2+sampling_frequency*max_time_gap/2;
    } else {
      Rcpp::warning("Block edge error");
    }
    if(i>0 and block_pos(i, 5)  <  block_pos(i-1, 6)) {
      //corrected egde for overlap
      if(block_pos(i-1, 15) >= 1 and block_pos(i, 15) < 1){
        block_pos(i, 5) = block_pos(i-1, 6)+1;
      } else if(block_pos(i-1, 15) < 1 and block_pos(i, 15) >= 1) {
        block_pos(i-1, 6) = block_pos(i, 5)-1;
      } else {
        int new_edge = block_pos(i-1, 4)+(block_pos(i, 4)-block_pos(i-1, 4))/2;
        block_pos(i, 5) = new_edge+1;
        block_pos(i-1, 6) = new_edge;
      }
    }
  }
  
  colnames(block_pos) = Rcpp::CharacterVector::create("onset",
           "end",
           "start_pulse",
           "end_pulse",
           "centre",
           "period_edge_l",
           "period_edge_r",
           "burst_stim",
           "block_isolation",
           "frequency",
           "sine",
           "pulse",
           "ramp_front",
           "ramp_end",
           "hyper_block",
           "hyper_block_frequency",
           "stim_length");
  return Rcpp::List::create(Rcpp::Named("PulseMat") = stim_mat,
                            Rcpp::Named("BlockMat") = block_pos);
}