stimulation_string <- function(pre, post, stimulation_block) {
  if(stimulation_block==0) { return("outside")}
  else if(pre==1){return("pre")}
  else if(post==1){return("post")}
  else if(stimulation_block==1 & pre==0 & post==0) {return("stimulation")}
}

stimulation_type_string <- function(pulse, sine, ramp_front, ramp_end) {
  if(pulse==1) { return("pulse")}
  else if(sine==1){return("sine")}
  else if(ramp_front==1){return("ramp_front")}
  else if(ramp_end==1) {return("ramp_end")}
  else {return("no_stim")}
}

cluster_type <- function(x, cluster_type, cluster_nr){
  return(as.character(cluster_type[cluster_nr==x]))
}