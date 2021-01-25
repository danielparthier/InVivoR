#include <Rcpp.h>

//' @title Spike stimulation properties
//' 
//' @description This function assigns stimulation properties to every spike so that it can be incorporated in a table.
//' 
//' @name SpikeStimProperties
//' @param SpikeIdx An integer vector representing the spike time point as sampling point (index).
//' @param StimMat A matrix with stimulation properties (from Pulse_Seq_correction).
//' @param BlockMat A matrix with stimulation block properties..
//' @param SamplingRate An integer indicating the sampling frequency.
//' @param Isolated A bool indicting whether isolated stimulation outside of blocks should be included.
//' 
//' @return Returns a matrix with stimulation properties for every spike ("onset", "peak_loc_offset", "peak_amp", "stim_length", "phase", "pulse", "sine", "ramp_front", "ramp_end", "frequency", "burst stim", "pulse_nr_block", "pulse_nr", "stimulation_block", "stimulation_timing", "pre", "post", "hyper_block", "hyper_block_frequency").
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix SpikeStimProperties(const Rcpp::IntegerVector& SpikeIdx,
                                          const Rcpp::NumericMatrix& StimMat,
                                          const Rcpp::NumericMatrix& BlockMat,
                                          const int& SamplingRate,
                                          const bool Isolated) {
  R_xlen_t mat_end = StimMat.nrow();
  Rcpp::NumericMatrix block_mat = Rcpp::clone(BlockMat);
  int actual_stim_count = 0;
  for(R_xlen_t j = 0; j < mat_end; ++j) {
    if((StimMat(j, 12) == 1 and StimMat(j, 10) != StimMat(j, 15)) or StimMat(j, 10) == StimMat(j, 15)) {
      ++actual_stim_count;
    }
  }
  
  // construct new matrix 
  Rcpp::NumericMatrix stim_mat(actual_stim_count,StimMat.ncol());
  int actual_stim_count_tmp = 0;
  int stim_count = 1;
  for(R_xlen_t j = 0; j < mat_end; ++j) {
    if((StimMat(j, 12) == 1 and StimMat(j, 10) != StimMat(j, 15)) or StimMat(j, 10) == StimMat(j, 15)) {
      if(j > 0 and StimMat(j, 16) != StimMat(j-1, 16)) {
        stim_count = 1;
        block_mat(StimMat(j, 16)-1, 2) = actual_stim_count_tmp+1;
        block_mat(StimMat(j, 16)-2, 3) = stim_mat(actual_stim_count_tmp-1, 13);
      } else if(j ==  mat_end-1 and StimMat(j, 16) == StimMat(j-1, 16)) {
        block_mat(StimMat(j, 16)-1, 3) = actual_stim_count_tmp+1;
      }
      stim_mat.row(actual_stim_count_tmp) = StimMat.row(j);
      stim_mat(actual_stim_count_tmp, 12) = stim_count;
      stim_mat(actual_stim_count_tmp, 13) = actual_stim_count_tmp+1;
      ++stim_count;
      ++actual_stim_count_tmp;
    }
  }
  
  R_xlen_t SpikeIdx_end = SpikeIdx.size();
  R_xlen_t stim_mat_end = stim_mat.nrow();
  R_xlen_t block_end = block_mat.nrow();
  R_xlen_t new_i = 0;
  R_xlen_t new_block_i = 0;
  int start_phase = 0;
  int end_phase = 0;
  
  Rcpp::NumericMatrix output_mat(SpikeIdx_end,20);
  for(R_xlen_t j = 0; j < SpikeIdx_end; ++j) {
    block_end = block_mat.nrow();
    for(R_xlen_t block_i = new_block_i; block_i < block_end; ++block_i) {
      if(block_mat(block_i, 5) <= SpikeIdx[j] and block_mat(block_i, 6) >= SpikeIdx[j]) {
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
        output_mat(j, 15) = (SpikeIdx[j]-block_mat(block_i, 0))/SamplingRate;
        if(SpikeIdx[j] < block_mat(block_i, 0)) {
          //pre
          output_mat(j, 16) = 1;
        } else if(SpikeIdx[j] > block_mat(block_i, 1)){
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
            if(SpikeIdx[j] >= start_phase and SpikeIdx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } else if(stim_mat(i, 12) == 1 and i < stim_mat_end) {
            //first and isolated pulse or last
            start_phase = block_mat(block_i, 0);
            end_phase = block_mat(block_i, 1);
            if(SpikeIdx[j] >= start_phase and SpikeIdx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } else if(stim_mat(i, 12) > 1 and i < stim_mat_end-1 and stim_mat(i+1, 16) == stim_mat(i, 16)) {
            //following pulses
            start_phase = stim_mat(i, 0)-(stim_mat(i, 0)-stim_mat(i-1, 0))/2;
            end_phase = stim_mat(i, 0)+(stim_mat(i+1, 0)-stim_mat(i, 0))/2;
            if(SpikeIdx[j] >= start_phase and SpikeIdx[j] <= end_phase) {
              new_i = i;
              goto data_entry;
            }
          } else if((stim_mat(i, 12) > 1 and i < stim_mat_end-1 and stim_mat(i+1, 16) != stim_mat(i, 16)) or (stim_mat(i, 12) > 1 and i == stim_mat_end-1)) {
            //last puls in block but not last in total
            start_phase = stim_mat(i, 0)-(stim_mat(i, 0)-stim_mat(i-1, 0))/2;
            end_phase = block_mat(block_i, 1);
            if(SpikeIdx[j] >= start_phase and SpikeIdx[j] <= end_phase) {
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
            output_mat(j, 2) = (SpikeIdx[j]-stim_mat(new_i, 0))/SamplingRate;
            //phase
            output_mat(j, 5) = 360.0*(SpikeIdx[j]-start_phase)/(end_phase-start_phase);
            // pulse number block
            output_mat(j, 12) = stim_mat(new_i, 12);
            // pulse number
            output_mat(j, 13) = stim_mat(new_i, 13);
          };
          //peak amp
          output_mat(j, 3) = stim_mat(new_i, 3);
          break;
      } else if(block_mat(block_i, 6) < SpikeIdx[j]){
        new_block_i = block_i+1;
        if(block_i+1 == block_end) {
          break;
        }
        break;
      }
    }
    output_mat(j, 0) = SpikeIdx[j];
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

