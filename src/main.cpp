/*
 * Copyright (C) 2017 by
 * 
 *  Yang Qinjuan
 *  yangqinj@mail2.sysu.edu.cn
 *  Computer and Data Science
 *  Sun Yat-sat University
 *
 */

#include <iostream>
#include "model.h"

int main(int argc, char const **argv) {
  Model model;
  if (model.InitEst(argc, argv)) {
    std::cout << "Error Init for Estimate!" << std::endl;
    return 1;
  }

  model.PrintConfig();

  model.Estimate();

  return 0;
}

