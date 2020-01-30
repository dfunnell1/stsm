/*
 * Copyright (C) 2017 by
 * 
 * Yang Qinjuan
 * yangqinj@mail2.sysu.edu.cn
 * Computer and Data Science
 * Sun Yat-sat University
 *
 */

#include "utils.h"
#include <algorithm>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::stoi;
using std::stof;

/**
 * Parse arguments from command line
 * @param  argc   Number of arguments
 * @param  argv   Strings of arguments 
 * @param  pmodel Pointer to the  model
 * @return        Whether parse successfuly. return 0 if it's success, otherwise 1.
 */
int Utils::ParseArgs(int argc, char const **argv, Model *pmodel) {
  int ttimes = -1;

  string doc_dir = ""; // directory of documentId file
  string lexicon_dir = ""; // directory of sentiment words files
  string output_dir = ""; // directory of output files

  double alpha = -1.0;
  double gamma = -1.0;
  string beta_str = "";
  double * betas = new double[3];

  int ntopics = -1;
  int nsentis = -1;
  int niters = -1;

  int i = 0;
  while (i < argc) {
    string arg = argv[i];

    if (arg == "-ntopics") {
      ntopics = stoi(argv[++i]);

    } else if (arg == "-nsentis") {
      nsentis = stoi(argv[++i]);

    } else if (arg == "-niters") {
      niters = stoi(argv[++i]);

    } else if (arg == "-betas") {
      beta_str = argv[++i];

    } else if (arg == "-doc_dir") {
      doc_dir = argv[++i];

    } else if (arg == "-lexicon_dir") {
      lexicon_dir = argv[++i];

    } else if (arg == "-output_dir") {
      output_dir = argv[++i];

    } else if (arg == "-alpha") {
      alpha = stof(argv[++i]);

    } else if (arg == "-gamma") {
      gamma = stof(argv[++i]);

    } else if (arg == "-ttimes") {
      ttimes = stoi(argv[++i]);
    }
    i++;
  }

  if (ttimes == -1) {
    cout << "Please specify the ttimes!" << endl;
    return 1;
  } else {
    pmodel->ttimes = ttimes;
  }

  if (doc_dir == "") {
    cout << "Please specify the directory of document!" << endl;
    return 1;

  } else {
    pmodel->doc_dir = doc_dir;
  }

  if (lexicon_dir == "") {
    cout << "Do not use lexicon!" << endl;

  } else {
    pmodel->lexicon_dir = lexicon_dir;
  }

  if (output_dir == "") {
    cout << "Please specify the directory of output files!" << endl;
    return 1;
  } else {
    pmodel->output_dir = output_dir;
  }

  if (ntopics > 0) {
    pmodel->ntopics = ntopics;
  }

  if (nsentis > 0) {
    pmodel->nsentis = nsentis;
  }

  if (niters > 0) {
    pmodel->niters = niters;
  }

  if (alpha > 0) {
    pmodel->alpha = alpha;
  } else {
    pmodel->alpha = 50.0 / pmodel->ntopics;
  }

  if (gamma > 0) {
    pmodel->gamma = gamma;
  }

  if (beta_str != "") {
    vector<string> beta_str_vec = Split(beta_str, "/");
    if (beta_str_vec.size() != 3) {
      cout << "The beta should be length of 3: Common/SentiWords/Other SentiWords" << endl;
      return 1;
    } else {
      for (int i = 0; i < 3; ++i)
      {
        pmodel->betas[i] = stof(beta_str_vec[i]);
      }
    }
  }
  
  return 0;
}

/**
 * Split string by delimiters.
 * @param text  text to be splited
 * @param delims delimeters to split text
 * @return tokens of text after splited 
 */
vector<string> Utils::Split(const string & text, const string & delims) {
  bool keep_empty = false;
    vector<string> result;
      if (delims.empty()) {
          result.push_back(text);
          return result;
      }
      string::const_iterator substart = text.begin(), subend;
      while (true) {
          subend = search(substart, text.end(), delims.begin(), delims.end());
          string temp(substart, subend);
          if (keep_empty || !temp.empty()) {
              result.push_back(temp);
          }
          if (subend == text.end()) {
              break;
          }
          substart = subend + delims.size();
      }
      return result;
}

/**
 * Sort probabilities of words and corresponding words at the same time.
 * @param probs probabilities
 * @param words ids of words
 */
void Utils::Sort(vector<double> & probs, vector<int> & words) {
  for (int i = 0; i < probs.size() - 1; i++) {
    for (int j = i + 1; j < probs.size(); j++) {
      if (probs[i] < probs[j]) {
        double tempprob = probs[i];
        int tempword = words[i];
        probs[i] = probs[j];
        words[i] = words[j];
        probs[j] = tempprob;
        words[j] = tempword;
      }
    }
  }
}

void Utils::QuickSort(vector<pair<int, double> > & vect, int left, int right) {
  int l_hold, r_hold;
  pair<int, double> pivot;
  
  l_hold = left;
  r_hold = right;    
  int pivotidx = left;
  pivot = vect[pivotidx];

  while (left < right) {
    while (vect[right].second <= pivot.second && left < right) {
      right--;
    }
    if (left != right) {
      vect[left] = vect[right];
      left++;
    }
    while (vect[left].second >= pivot.second && left < right) {
      left++;
    }
    if (left != right) {
      vect[right] = vect[left];
      right--;
    }
  }

  vect[left] = pivot;
  pivotidx = left;
  left = l_hold;
  right = r_hold;
  
  if (left < pivotidx) {
    QuickSort(vect, left, pivotidx - 1);
  }
  if (right > pivotidx) {
    QuickSort(vect, pivotidx + 1, right);
  }    
}



