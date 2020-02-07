#' Extract meta data from ND-Manager xml file
#' 
#' Function to extract meta data from xml file provided by the ND-Manager. Includes channel number,
#' sampling rate, spike groups.
#' @param FileName String to input directory including complete file name.
#'
#' @return Returns a list which includes the number of channels, sampling rate, number of spike groups, and spike groups, number of anatomical groups, and anatomical groups.
#' @export
MetaData <- function(FileName) {
  OutputList <- list()
  if(file.exists(FileName)) {
    xmlFile <- xml2::read_xml(FileName)
    xmlList <- xml2::as_list(xmlFile)
    OutputList$ChannelNr <- as.integer(unlist(xmlList$parameters$acquisitionSystem$nChannels))
    OutputList$SamplingRate <- as.integer(unlist(xmlList$parameters$acquisitionSystem$samplingRate))
    if(length(grep(pattern = "units", x = names(xmlList$parameters)))) {
      OutputList$SpikeGroupNumber <- length(xmlList$parameters$spikeDetection$channelGroups)
      OutputList$SpikeGroups <- sapply(X = 1:OutputList$SpikeGroupNumber,
                                       FUN = function(i) {SpikeOut <- as.integer(unlist(xmlList$parameters$spikeDetection$channelGroups[i]))
                                         return(SpikeOut[1:(length(SpikeOut)-3)])
                                         }
      )  
    }
    if(length(grep(pattern = "anatomicalDescription", x = names(xmlList$parameters)))) {
      OutputList$AnatomicalGroupNumber <- length(xmlList$parameters$anatomicalDescription$channelGroups)
      OutputList$AnatomicalGroups <- sapply(X = 1:OutputList$AnatomicalGroupNumber,
                                       FUN = function(i) {return(as.integer(unlist(xmlList$parameters$anatomicalDescription$channelGroups[i])))
                                         }
      )  
    }
    OutputList$ChannelMap$ChannelMap
    tmpChannels <- channel_clustering(channel_map = GetChanMap(dirname(path = FileName)), channel_cluster_dist = 50)
    OutputList$ChannelMap$Clusters <- data.frame(ChannelNr = tmpChannels$ChannelNr,
                                                 ChannelCluster = tmpChannels$ChannelCluster,
                                                 RefChannel = tmpChannels$RefChannel)
    OutputList$ChannelMap$ChannelMap <- data.frame(ChannelNr = tmpChannels$ChannelNr,
                                                   x_coord = tmpChannels$x_coord,
                                                   y_coord = tmpChannels$y_coord)
    return(OutputList) 
  } else {
    warning("Could not find xml file! \nPlease choose.", immediate. = T)
    if(dir.exists(dirname(FileName))) {
      MetaData(utils::choose.files(default = dirname(FileName)))
    } else {
      MetaData(utils::choose.files())
    }
  }
}
