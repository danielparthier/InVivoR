#include <RcppArmadillo.h>
#include <fstream>
#include <iostream>
#define ARMA_NO_DEBUG

// [[Rcpp::depends(RcppArmadillo)]]
//' Binary file access
//' 
//' This function extracts data points from a binary file based on time points and a window
//' which will cut out the data of and concatenates them based on the channels.
//' 
//' @param FILENAME A string as path to file with single "/".
//' @param spikePoints An integer vector for time points which correspond to sampling points.
//' @param WINDOW An integer indicating points taken before and after time point (default = 40).
//' @param CHANNELCOUNT An integer indicating number of channels in recording (default = 32).
//' @param CACHESIZE An integer indicating the size of cache used for buffering file in bytes (default = 512000).
//' @param BYTECODE An integer indicating number of bytes coding for bit rate (8bit = 1, 16bit = 2 etc.) (default = 2).
//' @return Returns an armadillo cube with extracted spikes.
//' @export
// [[Rcpp::export]]
arma::cube BinaryFileAccess(const std::string& FILENAME,
                            arma::vec& spikePoints,
                            const int& WINDOW = 40,
                            const unsigned int& CHANNELCOUNT = 32,
                            const unsigned int& CACHESIZE = 512000,
                            const unsigned int& BYTECODE = 2){
  // open stream
  std::ifstream f;
  f.open(FILENAME.c_str(), std::ios::binary | std::ios::in);
  //get file size
  f.seekg(0, std::ios::end);
  bool lastRun = false; 
  const long unsigned int FILESIZE = f.tellg();
  //find time points with intact WINDOW
  arma::vec spikePointsUse = spikePoints(arma::find(spikePoints>WINDOW and spikePoints<(FILESIZE-WINDOW)));
  long unsigned int spikePos = 0;
  const int& TOTALWINDOW = WINDOW*2+1;
  //compute BUFFERSIZE (multiple of channel count to assure intact time index for every channel)
  const unsigned int& BUFFERSIZE = round(CACHESIZE/CHANNELCOUNT)*CHANNELCOUNT;//TOTALWINDOW*CHANNELCOUNT;
  int length = BYTECODE*BUFFERSIZE;
  short fileBuf[BUFFERSIZE];
  //initiate output vector
  arma::cube outCube(CHANNELCOUNT, TOTALWINDOW, spikePointsUse.size(), arma::fill::zeros);
  //set start and end of first buffer
  long unsigned int bufferStart = spikePointsUse.at(0)-WINDOW;
  long unsigned int bufferEnd = bufferStart+BUFFERSIZE/CHANNELCOUNT;
  //run through file
  while(bufferEnd*BYTECODE*CHANNELCOUNT<FILESIZE and spikePos < spikePointsUse.size() and lastRun == false) {
    //buffer
    f.seekg(bufferStart*CHANNELCOUNT*BYTECODE, std::ios::beg);
    f.readsome((char*)&fileBuf, length);
    while(spikePos < spikePointsUse.size() and (spikePointsUse.at(spikePos)+WINDOW) < bufferEnd){
      int onset = (spikePointsUse.at(spikePos)-WINDOW)-bufferStart;
      //write from buffer to vector
      int windowIdx = 0;
      for(long unsigned int i = onset*CHANNELCOUNT; i < (onset+TOTALWINDOW)*CHANNELCOUNT; i+=CHANNELCOUNT) {
        for(unsigned int channelRun = 0; channelRun < CHANNELCOUNT; ++channelRun) {
          outCube.at(channelRun, windowIdx,spikePos) = fileBuf[i+channelRun]*0.195;
        }
        ++windowIdx;
      }
      ++spikePos;
    }
    bufferStart = bufferEnd-TOTALWINDOW;
    bufferEnd = bufferStart+BUFFERSIZE/CHANNELCOUNT;
    if(bufferEnd > FILESIZE/CHANNELCOUNT/BYTECODE) {
      bufferEnd = FILESIZE/CHANNELCOUNT/BYTECODE;
      length = (bufferEnd-bufferStart)*BYTECODE;
      lastRun = true;
    }
  }
  f.close();
  return outCube;
}