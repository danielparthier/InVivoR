#' Extract meta data from ND-Manager xml file
#' 
#' Function to extract meta data from xml file provided by the ND-Manager. Includes channel number,
#' sampling rate, spike groups.
#' @param FileName String to input directory including complete file name.
#'
#' @return
#' @export
#'
#' @examples
MetaData <- function(FileName) {
  OutputList <- list()
  if(file.exists(FileName)) {
    xmlFile <- xml2::read_xml(FileName)
    xmlList <- xml2::as_list(xmlFile)
    OutputList$ChannelNr <- as.integer(unlist(xmlList$parameters$acquisitionSystem$nChannels))
    OutputList$SamplingRate <- as.integer(unlist(xmlList$parameters$acquisitionSystem$samplingRate))
    OutputList$SpikeGroupNumber <- length(xmlList$parameters$spikeDetection$channelGroups)
    OutputList$SpikeGroups <- sapply(X = 1:OutputList$SpikeGroupNumber,
                                     FUN = function(i) {as.integer(unlist(xmlList$parameters$anatomicalDescription$channelGroups[i]))}
    )
    #OutputList$ChannelMap$ChannelMap
    #OutputList$ChannelMap$Clusters <- channel_clustering(channel_map = ChannelMap, channel_numbers = OutputList$ChannelNr, channel_cluster_dist = 50)
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
