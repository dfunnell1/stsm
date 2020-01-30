/*
 * Copyright (C) 2017 by
 * 
 *  Yang Qinjuan
 *  yangqinj@mail2.sysu.edu.cn
 *  Computer and Data Science
 *  Sun Yat-sat University
 *
 */

#ifndef _MODEL_H_
#define _MODEL_H_ 

#include <string>
#include "dataset.h"

using std::shared_ptr;

class Model {
 public:
  int ttimes; // t-th times

  // ----- model parameters
  int ntopics; // number of topics
  int nsentis; // number of sentiments
  int niters; // iteration steps

  double alpha;
  double gamma;
  double * betas;  // size = 3,  betas[0]: common word; betas[1]: current sentiment word; 
           // betas[2]: other sentiment word
  double ** beta; // size  = nsentis * nvocab
  
  double sum_alpha; // ntopics * alpha
  double sum_gamma; // nsentis * gamma
  double * sum_beta; // size = nsentis, sum_beta[i]: sum of beta[i] for sentiment i


  // ------ dataset and its setting
  string doc_dir; // directory of document ids and vocabulary
  string lexicon_dir; // directory of sentiment lexicon
  string output_dir; // directory of output files

  int nvocab; // vocabulary size

  shared_ptr<Dataset> ptrndat; // pointer to training dataset

  int ndocs; // number of (training) documents

  // model variables
  double ** theta; // topic distribution, size = (ndocs * ntopics)
  double *** pi; // sentiment distribution, size = (ndocs * nsentis)
  double *** phi; // word distribution for each topic and sentiment, 
          // size = (ntopics * nsentis * nvocab)
  double ** doc_senti_prob; // sentiment distribution for documents, D * S

  int ** nDT; // size = (ndocs * ntopics), nDT[i][j]: number of sentences in document i assigned to topic j
  int *** nDTS; // size = (ndocs * ntopics * nsentis), nDS[i][j][k]: number of segments 
        // in document i assigned to topic j and sentiment k
  int *** nTSW; // size = (ntopics * nsentis * nvocab)
         // nTSW[i][j][k]: how many times word k is assigned to topic i and sentiment j
  int * sumDT; // size = (ndocs), sumDT[i]: number of sentences in document i
  int ** sumDTS; // size = (ndocs), sumDS[i]: number of segments in document i
  int ** sumTSW; // size = (ntopics * nsentis), sumTSW[i][j]: total number of words 
          // assigned to topic i and sentiment j

  Model();
  ~Model();

  // initialize model parameter
  int InitEst(int argc, char const ** argv, bool randomize=false);

  // print model configuration ifnormation
  void PrintConfig();

  // estimate topic and sentiment assignment
  void Estimate(int interval = 100);
  void SamplingDoc(const shared_ptr<Document>&);
  void SamplingSentence(const shared_ptr<Sentence>&, int);
  void SamplingSeg(const shared_ptr<Segment>&, int, int);

  // sample a random int number from probability distribution
  int SampleNew(double * &, int );

  // compute model variables
  void ComputeTheta();
  void ComputePi();
  void ComputePhi();
  // compute probability for each sentiment within a document
  void ComputeDocSentiProb();

  // save variables and model parameters
  int SaveModel();
  int SaveModelTheta(const string&);
  int SaveModelPi(const string&);
  int SaveModelPhi(const string&);
  int SaveModelDocSentiProb(const string&);
  int SaveModelAssign(const string&);
  int SaveModelTopWords(const string&, int topk = 30);
};

#endif