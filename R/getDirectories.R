#' Get Directory
#'
#' This function will get the directory for any given label by opening a user interaction.
#' @param mainDirectory A string locating the path.
#' @param Labels A vector of strings including the label names.
#'
#' @return Returns a data.frame which includes the labels and the directory to the folders.
#' @export
getDirectory <- function(mainDirectory, Labels) {
  directoryFrame <- data.frame(row.names = Labels)
  for(i in 1:length(Labels)) {
    directoryFrame[i,1] <- Labels[i]
    directoryFrame[i,2] <- tcltk::tk_choose.dir(default = mainDirectory, caption = paste0("choose data folder for ", Labels[i]))
  }
  colnames(directoryFrame) <- c("Label", "dataPath")
  return(directoryFrame)
}

#' Get Data Directory
#'
#' This function will get the directory for any given label by opening a user interaction.
#' @param mainDirectory A string locating the path.
#' @param recLabels A vector of strings including the recording label names.
#' @param stimLabels A vector of strings including the stimulation label names.
#' @param stimRecIdentical A bool indicating whether stimulation and recording directory are identical.
#'
#' @return Returns a data.frame which includes the labels and the directory to the folders.
#' @export
getDataDirectory <- function(mainDirectory = NULL, recLabels = NULL, stimLabels = NULL, stimRecIdentical = NULL) {
  if(is.null(mainDirectory)) {
    warning("Missing directory")
  }
  if(is.null(recLabels) & is.null(stimLabels)) {
    warning("Missing labels: enter recLabels and/or stimLabels")
  }
  if(stimRecIdentical==T & (is.character(recLabels) || is.character(stimLabels))) {
    if(is.null(recLabels)) {
      recLabels <- stimLabels
    } 
    stimDf <- getDirectory(mainDirectory = mainDirectory, Labels = recLabels)
    return(list(stimulation = stimDf, recording = stimDf))
  } else if(is.character(recLabels) & is.character(stimLabels)) {
    if(is.null(stimRecIdentical) & all.equal(recLabels, stimLabels)) {
      stimDf <- getDirectory(mainDirectory = mainDirectory, Labels = recLabels)
      return(list(stimulation = stimDf, recording = stimDf))
    } else {
      stimDf <- getDirectory(mainDirectory = mainDirectory, Labels = stimLabels)
      recDf <- getDirectory(mainDirectory = mainDirectory, Labels = recLabels)
      return(list(stimulation = stimDf, recording = recDf))
    }
  } else if((stimRecIdentical==F & is.character(recLabels)) & is.null(stimLabels)) {
    warning("Only recording directory selected (no stimulation)")
    recDf <- getDirectory(mainDirectory = mainDirectory, Labels = recLabels)
    return(list(recording = recDf))
  } else {
    warning("missing parameters")
  }
}
