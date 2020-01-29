/*
 * Copyright (C) 2017 by
 * 
 * 	Yang Qinjuan
 *	yangqinj@mail2.sysu.edu.cn
 * 	Computer and Data Science
 * 	Sun Yat-sat University
 *
 */

#include <iostream>
#include "model.h"
using namespace std;



int main(int argc, char const **argv)
{
	
	Model model;
	if (model.InitEst(argc, argv)) {
		cout << "Error Init for Estimate!" << endl;
		return 1;
	}	

	// if (model.InitInf()) {
	// 	cout << "Error Init for Inference!" << endl;
	// 	return 1;
	// }

	model.PrintConfig();

	model.Estimate();
	// model.Inference();

	return 0;
}

