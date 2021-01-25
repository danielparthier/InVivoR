# load chanMap from directory and convert it to  x,y only data.frame
#' @title Get channel map
#' 
#' @description Function wrapper to load channel map from a given directory.
#' 
#' @name UnitChannelMap
#' @param SpikeIdx Indeces for spike times.
#' @param Units Vector having units.
#' @param FileLoc String with file name
#' @param SubSampleSize Subsample of all spikes.
#' @param WindowSize Size of the window to extract spikes.
#' @param ChannelCount Number of Channels in file.
#' 
#' @return Returns a data frame containing the coordinates of the channels.
#' @export
UnitChannelMap <- function(SpikeIdx,
                           Units,
                           FileLoc,
                           SubSampleSize,
                           WindowSize,
                           ChannelCount) {
  
  UniqueUnits <- unique(Units)
  Amplitude <- vector(mode = "numeric", length = length(UniqueUnits))
  ChannelNr <- vector(mode = "integer", length = length(UniqueUnits))
  SpikeShape <- array(dim = c(ChannelCount,WindowSize*2+1,length(UniqueUnits)))
  for(i in seq_along(UniqueUnits)) {
    SpikeShape[,,i] <- SpikeMed(BinaryFileAccess(FILENAME = FileLoc,
                                                 spikePoints = sort(sample(x = SpikeIdx[Units==UniqueUnits[i]], size = SubSampleSize)),
                                                 WINDOW = WindowSize,
                                                 CHANNELCOUNT = ChannelCount))
    ChannelAndAmp <- MaxChannel(SpikeShape[,,i])
    ChannelNr[i] <- ChannelAndAmp[1]
    Amplitude[i] <- ChannelAndAmp[2]
  }
  return(list(ChannelNr=ChannelNr, Amplitude=Amplitude, UnitNr=UniqueUnits, SpikeShape=SpikeShape))
}
