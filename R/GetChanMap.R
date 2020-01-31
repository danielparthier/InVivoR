# load chanMap from directory and convert it to  x,y only data.frame
#' Get channel map
#' 
#' Function wrapper to load channel map from a given directory.
#' @param directory String to input directory.
#' @param FileName Name of file without ".mat".
#'
#' @return
#' @export
#'
#' @examples
GetChanMap <- function(directory, FileName = "chanMap") {
  FileString <- paste0(directory,"/",FileName, ".mat")
  OutputFrame <- loadChanMap(FileString)
  return(OutputFrame)
}

#' Load channel map
#' Function to load channel
#' @param FileName Complete filename including path and file ending.
#'
#' @return
#' @export
#'
#' @examples
loadChanMap <- function(FileName) {
  if(file.exists(FileName)) {
    channelMap <- R.matlab::readMat(FileName)
    return(data.frame(x = as.vector(channelMap$xcoords), y = as.vector(channelMap$ycoords)))  
  } else {
    warning("Could not find Channel Map! \nPlease choose.", immediate. = T)
    if(dir.exists(dirname(FileName))) {
      loadChanMap(utils::choose.files(default = dirname(FileName)))
    } else {
      loadChanMap(utils::choose.files())
    }
  }
}
