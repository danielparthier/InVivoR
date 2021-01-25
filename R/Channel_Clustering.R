### function to cluster channels of the channel map
### input: channel_map which includes the x/y coordinates, channel_numbers starting at 0, channel_cluster dist which is the distance in µm used to cluster channels
#' @title Channel clustering
#' 
#' @description Function to extract electrode clusters from channel map.
#' 
#' @name channel_clustering
#' @param channel_map Data.frame with with x, y coordinates and channel numbers.
#' @param channel_cluster_dist Double indicating minimal distance between clusters.
#' 
#' @return Returns a list which includes the number of channels, sampling rate, number of spike groups, and spike groups, number of anatomical groups, and anatomical groups.
#' @export
channel_clustering <- function(channel_map, channel_cluster_dist) {
  channel_numbers <- channel_map$ChanInd-1
  channel_out_vec <- vector(mode = "integer", length = length(channel_numbers))
  channel_cluster <- dbscan::dbscan(x = channel_map, eps = channel_cluster_dist, borderPoints = F, minPts = 2)
  for(i in 0:max(channel_cluster$cluster)) {
    channel_in_cluster <- channel_numbers[channel_cluster$cluster==i]
    if(i > 0) {
      channel_coord <- channel_map[channel_cluster$cluster==i,]
      mean_coord <- c(mean(channel_coord[,1]), mean(channel_coord[,2]))
      distance_in_cluster <- sqrt((channel_coord[,1]-mean_coord[1])^2 + (channel_coord[,2]-mean_coord[2])^2)
      channels_selection <- channel_in_cluster[distance_in_cluster==min(distance_in_cluster)]
      if(length(channels_selection)>1) {
        coord_selection <- channel_coord[channel_in_cluster %in% channels_selection,]
        min_x <- min(coord_selection[,1])
        channels_x_min <- channels_selection[min_x==coord_selection[,1]]
        if(length(channels_x_min) > 1) {
          min_y <- min(coord_selection[,2])
          channels_y_min <- channels_selection[min_y==coord_selection[,2]]
          channel_out <- channels_y_min
          if(length(channels_y_min) > 1) {
            warning("channels have same coordinates")
            channel_out <- channels_y_min[1]
          }
        } else {
          channel_out <- channels_x_min
        }
      }
    } else if (i == 0) {
      channel_out <- channel_numbers[channel_cluster$cluster==0]
    }
    channel_out_vec[channel_numbers %in% channel_in_cluster] <- channel_out
  }
  return(data.frame(ChannelNr = channel_map$ChanInd, ChannelCluster = channel_cluster$cluster, RefChannel = channel_out_vec+1, x_coord = channel_map[,1], y_coord = channel_map[,2]))
}
