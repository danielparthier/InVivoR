//#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <fstream>
#include <iostream>

// [[Rcpp::depends(RcppArmadillo)]]


// Armadillo implementation to convert digital input to binary output matrix(n,channels).
//' @title Convert digital inputs.
//' 
//' @description This function converts digital inputs of the intan system and separates the channels into a matrix.
//' 
//' @name convertToBinary
//' @param StimTrace  vector. 
//' @return Returns a list with the matrix, the channel ID and the label "Digital".
//' @export
// [[Rcpp::export]]
Rcpp::List convertToBinary(const arma::vec StimTrace){
  unsigned long long int StimLength = StimTrace.size();
  arma::vec UniqueValues = arma::nonzeros(arma::unique(StimTrace));
  arma::vec Stim = StimTrace;
  arma::vec DigIn = arma::sort(arma::intersect(arma::log2(UniqueValues), arma::floor(arma::log2(UniqueValues))), "descend");
  if(DigIn.size()>16) {
    Rcpp::stop("Found more than 16 digital signatures. Please Check if file is digital input.");
  }
  int upperIdx = DigIn.n_elem-1;
  arma::imat Output(StimTrace.size(), DigIn.size(), arma::fill::zeros);
  for(unsigned int i = 0; i<DigIn.n_elem; ++i) {
    int twoPow = std::pow(2, DigIn.at(i));
    arma::uvec rowtmp = arma::find(Stim>=twoPow);
    Stim.elem(rowtmp) -= twoPow;
    if((upperIdx-i)>0) {
      rowtmp += (upperIdx-i)*StimLength;
    }
    Output.elem(rowtmp).ones();
    Stim(arma::find(Stim<0)).zeros();
  }
  return Rcpp::List::create(Rcpp::Named("Output") = arma::conv_to<arma::dmat>::from(Output),
                            Rcpp::Named("ActiveChannels") = arma::sort(DigIn)+1,
                            Rcpp::Named("Type") = "Digital");
}


//' @title Binary file access
//' 
//' @description This function extracts data points from a binary file based on time points and a window
//' which will cut out the data of and concatenates them based on the channels.
//' 
//' @name BinaryFileAccess
//' @param FILENAME A string as path to file with single "/".
//' @param spikePoints An integer vector for time points which correspond to sampling points.
//' @param WINDOW An integer indicating points taken before and after time point (default = 40).
//' @param CHANNELCOUNT An integer indicating number of channels in recording (default = 32).
//' @param CACHESIZE An integer indicating the size of cache used for buffering file in bytes (default = 512000).
//' @param BYTECODE An integer indicating number of bytes coding for bit rate (8bit = 1, 16bit = 2 etc.) (default = 2).
//' 
//' @return Returns an armadillo cube with extracted spikes.
//' @export
// [[Rcpp::export]]
arma::cube BinaryFileAccess(const std::string& FILENAME,
                            arma::vec& spikePoints,
                            const int& WINDOW = 40,
                            const unsigned int& CHANNELCOUNT = 32,
                            const unsigned int& CACHESIZE = 51200,
                            const unsigned int& BYTECODE = 2){
  // open stream
  std::ifstream f;
  f.open(FILENAME.c_str(), std::ios::binary | std::ios::in);
  //get file size
  f.seekg(0, std::ios::end);
  bool lastRun = false; 
  const long long unsigned int FILESIZE = f.tellg();
  //find time points with intact WINDOW
  arma::vec spikePointsUse = spikePoints(arma::find(spikePoints>WINDOW and spikePoints<(FILESIZE/2-WINDOW)));
  long long unsigned int spikePos = 0;
  const int& TOTALWINDOW = WINDOW*2+1;
  //compute BUFFERSIZE (multiple of channel count to assure intact time index for every channel)
  const unsigned int& BUFFERSIZE = CACHESIZE;//round(CACHESIZE/CHANNELCOUNT)*CHANNELCOUNT;//TOTALWINDOW*CHANNELCOUNT;
  int length = BYTECODE*BUFFERSIZE;
  short fileBuf[CACHESIZE];//[BUFFERSIZE];
  //initiate output vector
  arma::cube outCube(CHANNELCOUNT, TOTALWINDOW, spikePointsUse.size(), arma::fill::zeros);
  //set start and end of first buffer
  long long unsigned int bufferStart = spikePointsUse.at(0)-WINDOW;
  long long unsigned int bufferEnd = bufferStart+BUFFERSIZE/CHANNELCOUNT;
  //run through file
  while(bufferEnd*BYTECODE*CHANNELCOUNT<FILESIZE and spikePos < spikePointsUse.size() and lastRun == false) {
    //buffer
    f.seekg(bufferStart*CHANNELCOUNT*BYTECODE, std::ios::beg);
    f.readsome((char*)&fileBuf, length);
    while(spikePos < spikePointsUse.size() and (spikePointsUse.at(spikePos)+WINDOW) < bufferEnd){
      int onset = (spikePointsUse.at(spikePos)-WINDOW)-bufferStart;
      //write from buffer to vector
      int windowIdx = 0;
      for(long long unsigned int i = onset*CHANNELCOUNT; i < (onset+TOTALWINDOW)*CHANNELCOUNT; i+=CHANNELCOUNT) {
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

//' @title Load stimulation file
//' 
//' @description This function extracts data points from a binary file either the analogin.dat or digitalin.dat generated intan system.
//' 
//' @name StimFileRead
//' @param FILENAME A string as path to file with single "/".
//' @param digital A bool indicating whether the file is a digital or analogin input (default = false).
//' 
//' @return Returns an armadillo cube with extracted spikes.
//' @export
// [[Rcpp::export]]
Rcpp::List StimFileRead(const std::string& FILENAME,
                                 const bool digital = false){
  // open stream
  std::ifstream f;
  f.open(FILENAME.c_str(), std::ios::binary | std::ios::in);
  // determine size
  f.seekg (0, std::ios::end);
  unsigned long long int length = f.tellg();
  f.seekg (0, std::ios::beg);
  // generate vector and read unsigned integers
  std::vector<unsigned short> val(length/sizeof(unsigned short));
  f.read((char*)&val[0], length);
  // close stream
  f.close();
  // convert std::vector to Rcpp::NumericVector
  if(digital) {
   // Rcpp::NumericVector Output = Rcpp::wrap(val);
    return convertToBinary(arma::conv_to<arma::vec>::from(val));
  } else {
    Rcpp::NumericVector Output = Rcpp::wrap(val);
    return Rcpp::List::create(Rcpp::Named("Output") = Output,
                              Rcpp::Named("ActiveChannels") = 1,
                              Rcpp::Named("Type") = "Analog"); 
  }
}

//' @title Load amplifier amplitude file
//' 
//' @description This function extracts data points from a binary file (amplifier.dat) generated by the intan system.
//' 
//' @name AmpFileRead
//' @param FILENAME A string as path to file with single "/".
//' @param ChannelNumber An integer indicating number of channels in recording (default = 32).
//' 
//' @return Returns an armadillo matrix (rows = ChannelNumber, columns = time).
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix AmpFileRead(const std::string& FILENAME,
                                const int ChannelNumber = 32){
  // open stream
  std::ifstream f;
  f.open(FILENAME.c_str(), std::ios::binary | std::ios::in);
  // determine size
  f.seekg (0, std::ios::end);
  unsigned long int length = f.tellg();
  f.seekg (0, std::ios::beg);
  // generate vector and read unsigned integers
  std::vector<short> val(length/sizeof(short));
  f.read((char*)&val[0], length);
  // close stream
  f.close();
  // convert std::vector to arma::mat with channels as rows
  //arma::vec outMat = arma::conv_to<arma::vec>::from(val);
  //outMat.reshape(length/sizeof(short)/ChannelNumber,ChannelNumber);
  Rcpp::NumericMatrix outMat = Rcpp::NumericMatrix(ChannelNumber, length/sizeof(short)/ChannelNumber,val.begin());
  return outMat*0.195; 
}



arma::mat AmpFileReadMerge(const std::string& FILENAME1,
                           const std::string& FILENAME2,
                           const int ChannelNumber = 32){
  // open stream
  std::ifstream f;
  f.open(FILENAME1.c_str(), std::ios::binary | std::ios::in);
  // determine size
  f.seekg (0, std::ios::end);
  unsigned long long int length = f.tellg();
  f.seekg (0, std::ios::beg);
  // generate vector and read unsigned integers
  std::vector<short> val(length/sizeof(short));
  f.read((char*)&val[0], length);
  // close stream
  f.close();
  arma::mat outMat1 = arma::conv_to<arma::mat>::from(val);
  outMat1.reshape(ChannelNumber,length/sizeof(short)/ChannelNumber);
  
  f.open(FILENAME2.c_str(), std::ios::binary | std::ios::in);
  // determine size
  f.seekg (0, std::ios::end);
  length = f.tellg();
  f.seekg (0, std::ios::beg);
  // generate vector and read unsigned integers
  f.read((char*)&val[0], length);
  // close stream
  f.close();
  
  // convert std::vector to arma::mat with channels as rows
  arma::mat outMat2 = arma::conv_to<arma::mat>::from(val);
  outMat2.reshape(ChannelNumber,length/sizeof(short)/ChannelNumber);
  
  return arma::join_rows(outMat1,outMat2)*0.195; 
}


// Rcpp::List convertToBinary(arma::vec x){
//   int activeChannels = 0;
//   int startBit = std::floor(std::log2(arma::max(x)));
//   arma::vec DigIn = arma::linspace(0,15,16);
//   arma::vec BitCheck = arma::zeros<arma::vec>(16);
//   arma::vec DigUsed = arma::zeros<arma::vec>(16);
//   for(unsigned int i = 0; i<DigIn.n_elem; ++i) {
//     BitCheck.at(i) = std::pow(2, DigIn.at(i));
//   }
//   arma::vec UniqueChannels = arma::unique(x);
//   for(int i = startBit; i>-1; --i) {
//     arma::vec remainder = arma::floor(UniqueChannels/BitCheck.at(i)-1);
//     arma::uvec foundIdx = arma::find(remainder == 0);
//     if(foundIdx.n_elem > 0) {
//       ++activeChannels;
//       DigUsed.at(i) = 1;
//     }
//     UniqueChannels.elem(foundIdx) -= BitCheck.at(i);
//   }
//   arma::uvec ChannelsUsed = arma::find(DigUsed>0);
//   arma::mat Output = arma::mat(x.n_elem, activeChannels, arma::fill::zeros);
//   arma::uvec ActiveTimes = arma::find(x>0);
//   for(int i = activeChannels-1; i>-1; --i) {
//     arma::vec remainder = arma::floor(x/BitCheck.at(ChannelsUsed.at(i))-1);
//     arma::vec tmp = arma::zeros<arma::vec>(x.n_elem);
//     arma::uvec foundIdx = arma::find(remainder == 0);
//     if(foundIdx.n_elem > 0) {
//       ++activeChannels;
//       DigUsed.at(i) = 1;
//     }
//     tmp.elem(foundIdx).ones();
//     x.elem(foundIdx) -= BitCheck.at(ChannelsUsed.at(i));
//     Output.col(i) = tmp;
//   }
//   return Rcpp::List::create(Rcpp::Named("Output") = Output,
//                             Rcpp::Named("ActiveChannels") = ChannelsUsed);
// }
// 
// 
// // [[Rcpp::export]]
// arma::mat MatCheck(){
//   arma::mat outMat(10,2);
//   arma::uvec ind = arma::regspace<arma::uvec>(0,19);
//   outMat.elem(ind) = arma::regspace(0,19);
//   return outMat;
// }


// work on Rcpp implementation to have logical matrix output (memory efficient)
//// [[Rcpp::export]]
// Rcpp::List convertToBinaryLogical(Rcpp::IntegerVector x){
//   int activeChannels = 0;
//   int MaxInt = max(x);
//   int startBit = std::floor(std::log2(MaxInt));
//   Rcpp::IntegerVector DigIn = Rcpp::seq(0,15);
//   Rcpp::IntegerVector UniqueChannels = unique(x);
//   
//   return Rcpp::List::create(Rcpp::Named("Output") = 1,
//                             Rcpp::Named("ActiveChannels") = 2);
// }