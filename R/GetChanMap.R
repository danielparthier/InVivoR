# load chanMap from directory and convert it to  x,y only data.frame
#' @title Get channel map
#' 
#' @description Function wrapper to load channel map from a given directory.
#' 
#' @name GetChanMap
#' @param directory String to input directory.
#' @param FileName Name of file without ".mat".
#' 
#' @return Returns a data frame containing the coordinates of the channels.
#' @export
GetChanMap <- function(directory, FileName = "chanMap") {
  FileString <- paste0(directory,"/",FileName, ".mat")
  OutputFrame <- loadChanMap(FileString)
  return(OutputFrame)
}

#' @title Load channel map
#' 
#' @description Function to load channel.
#' 
#' @name loadChanMap
#' @param FileName Complete filename including path and file ending.
#' 
#' @return Returns a data frame containing the coordinates of the channels.
#' @export
loadChanMap <- function(FileName) {
  if(file.exists(FileName)) {
    channelMap <- R.matlab::readMat(FileName)
    return(data.frame(x = as.vector(channelMap$xcoords), y = as.vector(channelMap$ycoords), ChanInd = as.vector(channelMap$chanMap)))  
  } else {
    warning("Could not find Channel Map! \nPlease choose.", immediate. = T)
    if(dir.exists(dirname(FileName))) {
      loadChanMap(utils::choose.files(default = dirname(FileName)))
    } else {
      loadChanMap(utils::choose.files())
    }
  }
}
