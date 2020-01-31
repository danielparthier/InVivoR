#include <Rcpp.h>

//' Spike stimulation properties
//' 
//' This function assigns stimulation properties to every spike so that it can be incorporated in a table.
//' 
//' @param spike_idx An integer vector representing the spike time point as sampling point (index).
//' @param stim_mat_org A matrix with stimulation properties (from Pulse_Seq_correctrion).
//' @param block_mat_org A matrix with stimulation block properties..
//' @param sampling_rate An integer indicating the sampling frequency.
//' @param include_isolated A bool indicting whether isolated stimulation outside of blocks should be included.
//' @return Returns a matrix with stimulation properties for every spike ("onset", "peak_loc_offset", "peak_amp", "stim_length", "phase", "pulse", "sine", "ramp_front", "ramp_end", "frequency", "burst stim", "pulse_nr_block", "pulse_nr", "stimulation_block", "stimulation_timing", "pre", "post", "hyper_block", "hyper_block_frequency").
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix spike_stim_properties(const Rcpp::IntegerVector& spike_idx,
                                          const Rcpp::NumericMatrix& stim_mat_org,
                                          const Rcpp::NumericMatrix& block_mat_org,
                                          const int& sampling_rate,
                                          const bool include_isolated) {
  R_xlen_t mat_end = stim_mat_org.nrow();
  Rcpp::NumericMatrix block_mat = Rcpp::clone(block_mat_org);
  int actual_stim_count = 0;
  for(R_xlen_t j = 0; j < mat_end; ++j) {
    if((stim_mat_org(j, 12) == 1 and stim_mat_org(j, 10) != stim_mat_org(j, 15)) or stim_mat_org(j, 10) == stim_mat_org(j, 15)) {
      ++actual_stim_count;
    }
  }
  
  // construct new matrix 
  Rcpp::NumericMatrix stim_mat(actual_stim_count,stim_mat_org.ncol());
  int actual_stim_count_tmp = 0;
  int stim_count = 1;
  for(R_xlen_t j = 0; j < mat_end; ++j) {
    if((stim_mat_org(j, 12) == 1 and stim_mat_org(j, 10) != stim_mat_org(j, 15)) or stim_mat_org(j, 10) == stim_mat_org(j, 15)) {
      if(j > 0 and stim_mat_org(j, 16) != stim_mat_org(j-1, 16)) {
        stim_count = 1;
        block_mat(stim_mat_org(j, 16)-1, 2) = actual_stim_count_tmp+1;
        block_mat(stim_mat_org(j, 16)-2, 3) = stim_mat(actual_stim_count_tmp-1, 13);
      } else if(j ==  mat_end-1 and stim_mat_org(j, 16) == stim_mat_org(j-1, 16)) {
        block_mat(stim_mat_org(j, 16)-1, 3) = actual_stim_count_tmp+1;
      }
      stim_mat.row(actual_stim_count_tmp) = stim_mat_org.row(j);
      stim_mat(actual_stim_count_tmp, 12) = stim_count;
      stim_mat(actual_stim_count_tmp, 13) = actual_stim_count_tmp+1;
      ++stim_count;
      ++actual_stim_count_tmp;
    }
  }
  
  R_xlen_t spike_idx_end = spike_idx.size();
  R_xlen_t stim_mat_end = stim_mat.nrow();
  R_xlen_t block_end = block_mat.nrow();
  R_xlen_t new_i = 0;
  R_xlen_t new_block_i = 0;
  int start_phase = 0;
  int end_phase = 0;
  
  Rcpp::NumericMatrix output_mat(spike_idx_end,20);
  for(R_xlen_t j = 0; j < spike_idx_end; ++j) {
    block_end = block_mat.nrow();
    for(R_xlen_t block_i = new_block_i; block_i < block_end; ++block_i) {
      if(block_mat(block_i, 5) <= spike_idx[j] and block_mat(block_i, 6) >= spike_idx[j]) {
        //spike is in block
        //put block information to spike
        //stim_length
        output_mat(j, 4) = block_mat(block_i, 16);
        //pulse
        output_mat(j, 6) = block_mat(block_i, 11);
        //sine
        output_mat(j, 7) = block_mat(block_i, 10);
        //ramp front
        output_mat(j, 8) = block_mat(block_i, 12);
        //ramp_end
        output_mat(j, 9) = block_mat(block_i, 13);
        //frequency
        output_mat(j, 10) = block_mat(block_i, 9);
        //burst stim
        output_mat(j, 11) = block_mat(block_i, 9) != block_mat(block_i, 15);
        //stimulation_block
        output_mat(j, 14) = 1;
        //stimulation_timing
        output_mat(j, 15) = (spike_idx[j]-block_mat(block_i, 0))/sampling_rate;
        if(spike_idx[j] < block_mat(block_i, 0)) {
          //pre
          output_mat(j, 16) = 1;
        } else if(spike_idx[j] > block_mat(block_i, 1)){
          //post
          output_mat(j, 17) = 1;
        } 
        //hyper_block
        output_mat(j, 18) = block_mat(block_i, 14);
        //hyper_block_frequency
        output_mat(j, 19) = block_mat(block_i, 15);
        new_block_i = block_i;
        for(R_xlen_t i = new_i; i < stim_mat_end; ++i) {
          if(stim_mat(i, 12) == 1 and i < stim_mat_end-1 and stim_mat(i+1, 16) == stim_mat(i, 16)) {
            //cases for first pulse and not last or isolated pulses
            start_phase = block_mat(block_i, 0);
            end_phase = stim_mat(i, 0)+(stim_mat(i+1, 0)-stim_mat(i, 0))/2;
            if(spike_idx[j] >= start_phase and spike_idx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } else if(stim_mat(i, 12) == 1 and i < stim_mat_end) {
            //first and isolated pulse or last
            start_phase = block_mat(block_i, 0);
            end_phase = block_mat(block_i, 1);
            if(spike_idx[j] >= start_phase and spike_idx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } else if(stim_mat(i, 12) > 1 and i < stim_mat_end-1 and stim_mat(i+1, 16) == stim_mat(i, 16)) {
            //following pulses
            start_phase = stim_mat(i, 0)-(stim_mat(i, 0)-stim_mat(i-1, 0))/2;
            end_phase = stim_mat(i, 0)+(stim_mat(i+1, 0)-stim_mat(i, 0))/2;
            if(spike_idx[j] >= start_phase and spike_idx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } else if((stim_mat(i, 12) > 1 and i < stim_mat_end-1 and stim_mat(i+1, 16) != stim_mat(i, 16)) or (stim_mat(i, 12) > 1 and i == stim_mat_end-1)) {
            //last puls in block but not last in total
            start_phase = stim_mat(i, 0)-(stim_mat(i, 0)-stim_mat(i-1, 0))/2;
            end_phase = block_mat(block_i, 1);
            if(spike_idx[j] >= start_phase and spike_idx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } 
        }
        data_entry:
          if(output_mat(j ,16) == output_mat(j ,17)) {
            //onset
            output_mat(j, 1) = stim_mat(new_i, 0);
            //time to peak
            output_mat(j, 2) = (spike_idx[j]-stim_mat(new_i, 0))/sampling_rate;
            //phase
            output_mat(j, 5) = 360.0*(spike_idx[j]-start_phase)/(end_phase-start_phase);
            // pulse number block
            output_mat(j, 12) = stim_mat(new_i, 12);
            // pulse number
            output_mat(j, 13) = stim_mat(new_i, 13);
          };
          //peak amp
          output_mat(j, 3) = stim_mat(new_i, 3);
          break;
      } else if(block_mat(block_i, 6) < spike_idx[j]){
        new_block_i = block_i+1;
        if(block_i+1 == block_end) {
          break;
        }
        break;
      }
    }
    output_mat(j, 0) = spike_idx[j];
  }
  colnames(output_mat) = Rcpp::CharacterVector::create("time_stamp",
           "onset",
           "peak_loc_offset",
           "peak_amp",
           "stim_length",
           "phase",
           "pulse",
           "sine",
           "ramp_front",
           "ramp_end",
           "frequency",
           "burst stim",
           "pulse_nr_block",
           "pulse_nr",
           "stimulation_block",
           "stimulation_timing",
           "pre",
           "post",
           "hyper_block",
           "hyper_block_frequency");
  return output_mat;
}

