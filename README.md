# SegmentLevelJointTopicSentimentModel

A C/C++ implementation of Segment-Level Joint Topic-Sentiment Model (STSM) using Gibbss Sampling for parameter estimation. 

Yang Q, Rao Y, Xie H, et al. Segment-Level Joint Topic-Sentiment Model for Online Review Analysis[J]. IEEE Intelligent Systems, 2019, 34(1): 43-50.

## Compile

A C/C++11.0 compiler and the STL library. In the  Makefile, we use g++ as the default compiler command, if the C/C++ compiler on your system has another name, you can modify the CC variable in the Makefile.
	
Go to home directory and type:

```bash
$ make clean
$ make all
```

## Usage

```bash
$ main [-ntopics <int>] [-niters <int>] [-nsentis <int>] -doc_dir <string> -output_dir <string> [-lexicon_dir <string>]
```

in which (parameters in `[]` are optional):

+ -ntopics <int>:
  	The number of topics. Default value is 100.

+ -niters <int>:
  	The iteration steps. Default value is 1000.

+ -nsentis <int>:
  	The number of sentiments. Default value is 2.

+ -doc_dir <string>:
  	Directory of input training data file, where contains two necessary files. The first file name is "DocumentId.txt", which is the ids for each documents. Each line in this file is a document where sentences are split by two tabs <\t\t>, segments are split by one tab <\t> and words in segment are split by a space <space>. The second file name is "vocabulary.txt", which is the vocabulary word string. Each line inside this file is a word string.

+ -lexicon_dir <string>:
  	Directory of the sentiment lexicon files. Each file has the name "SentiWords_x" where x is 0,1,2... Each line is a word.

+ -output_dir <string>:
  	Directory of the output files. 

  + assignment_<model_name>: topic and sentiment assignment for every documents.

  + phi_<model_name>: topic-sentiment-word distribution

  + pi_<model_name>: document-sentiment distribution

  + theta_<model_name>: document-topic distribution

  + top<int>_words_<model_name>: top k words in each topic-sentiment

  + model_name is in the format: 

    `ntopics<int>_nsentis<int>(<int>)_niters<int>_alpha<double>gamma<double>_betas<double/double/double>`

    nsentis<int>(<int>): first int is the setting nsentis for model parameter, and second int represents how many sentiments are used for lexicon. 