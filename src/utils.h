/*
 * Copyright (C) 2017 by
 * 
 * Yang Qinjuan
 * yangqinj@mail2.sysu.edu.cn
 *  Computer and Data Science
 *  Sun Yat-sat University
 *
 */

#ifndef _UTILS_H_
#define _UTILS_H_

#include <vector>
#include "model.h"

using std::pair;

class Utils {
 public:
  Utils();
  ~Utils();

  // parse command line arguments
  static int ParseArgs(int argc, char const **argv, Model *pmodel);

  //split a string by a seperator
  static vector<string> Split(const string &text, const string &sep);

  // sort    
  static void Sort(vector<double> &probs, vector<int> &words);
  static void QuickSort(vector<pair<int, double>> &vect, int left, int right);  
};

#endif
