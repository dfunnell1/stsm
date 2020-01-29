/*
 * Copyright (C) 2017 by
 * 
 * 	Yang Qinjuan
 *	yangqinj@mail2.sysu.edu.cn
 * 	Computer and Data Science
 * 	Sun Yat-sat University
 *
 */


#include "model.h"
#include "utils.h"
#include "dataset.h"
#include <fstream>
#include <iostream>    
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <set>
using namespace std;

/**
 * Construtor of Model and set default parameters
 */
Model::Model() {

	this->ttimes = 0;

	this->ntopics = 100;
	this->nsentis = 2;
	this->niters = 1000;

	this->alpha = 50.0 / this->ntopics;
	this->gamma = 1.0;
	this->betas = new double[3];
	this->betas[0] = 0.001; // common word
	this->betas[1] = 0.1; // sentiment word
	this->betas[2] = 0.0; // other sentiment word
	
	this->beta = NULL;
	this->sum_alpha = this->alpha * this->ntopics;
	this->sum_gamma = this->gamma * this->nsentis;
	this->sum_beta = NULL;
	

	this->doc_dir = "";
	this->lexicon_dir = "";
	this->output_dir = "";

	this->ptrndat = NULL;
	this->ndocs = 0;
	this->nvocab = 0;


	this->theta = NULL;
	this->pi = NULL;
	this->phi = NULL;

	this->nDT = NULL;
	this->nDTS = NULL;
	this->nTSW = NULL;
	this->sumDT = NULL;
	this->sumDTS = NULL;
	this->sumTSW = NULL;

}


Model::~Model() {
	if (this->betas) {
		delete this->betas;
	}


	if (this->beta) {
		for (int s = 0; s < this->nsentis; ++s)
		{
			if (this->beta[s]) {
				delete this->beta[s];
			}
		}
	}

	if (this->sum_beta) {
		delete this->sum_beta;
	}

	if (this->ptrndat) {
		delete this->ptrndat;
	}

	if (this->theta) {
		for (int i = 0; i < this->ndocs; ++i)
		{
			delete this->theta[i];
		}
	}

	if(this->pi) {
		for (int i = 0; i < this->ndocs; ++i) {
			if (this->pi[i]) {
				for (int j = 0; j < this->ntopics; ++j)
				{
					delete this->pi[i][j];
				}
			}
		}
	}

	if (this->phi) {
		for (int i = 0; i < this->ntopics; ++i)
		{
			if (this->phi[i]) {
				for (int j = 0; j < this->nsentis; ++j)
				{
					delete this->phi[i][j];
				}
			}
		}
	}


	if (this->doc_senti_prob) {
		for (int d = 0; d < this->ndocs; ++d)
		{
			if (this->doc_senti_prob[d]) {
				delete doc_senti_prob[d];
			}
		}
	}

	if (this->nDT) {
		for (int i = 0; i < this->ndocs; ++i)
		{
			delete this->nDT[i];
		}
	}

	if (this->nDTS) {
		for (int i = 0; i < this->ndocs; ++i)
		{
			if (this->nDTS[i]) 
			{
				for (int j = 0; j < this->ntopics; ++j)
				{
					delete this->nDTS[i][j];
				}
			}
		}
	}

	if (this->nTSW) {
		for (int i = 0; i < this->ntopics; ++i)
		{
			if (this->nTSW[i]) {
				for (int j = 0; j < this->nsentis; ++j)
				{
					delete this->nTSW[i][j];
				}
			}
		}
	}


	if (this->sumDT) {
		delete this->sumDT;
	}

	if (this->sumDTS) {
		for (int i = 0; i < this->ndocs; ++i)
		{
			if (this->sumDTS[i])
				delete this->sumDTS[i];
		}
	}

	if (this->sumTSW) {
		for (int i = 0; i < this->ntopics; ++i)
		{
			if (this->sumTSW[i])
				delete this->sumTSW[i];
		}
	}
}


	
/**
 * Initialize model for estimation 
 * @param  argc mnumber of parameters
 * @param  argv parameters string
 * @return      whether success
 */
int Model::InitEst(int argc, char const ** argv, bool randomize) {

	if (Utils::ParseArgs(argc, argv, this)) {
		cout << "Error in parse arguments!" << endl;
		return 1;
	} 


	this->sum_gamma = this->gamma * this->nsentis;
	this->sum_alpha = this->alpha * this->ntopics;

	//********* Read dataset 
	ptrndat = new Dataset();
	ptrndat->SetNumSentis(this->nsentis);

	// read vocabulary from file whose name is vocabulary.txt
	if (ptrndat->ReadVocab(this->doc_dir)) {
		cout << "Error in read vocabulary!" << endl;
		return 1;
	}

	// read sentiment lexicon from files, where each file
	// is a word list for a sentiment and the file name is SentiWords_X.txt
	if (ptrndat->ReadLexicon(this->lexicon_dir)) {
		cout << "Error in read lexicon!" << endl;
		return 1;
	}

	if (ptrndat->ReadDocs(this->doc_dir  + "/DocumentId.txt")) {
		cout << "Error in read documents!" << endl;
		return 1;
	}

	this->ndocs = this->ptrndat->GetNumDocs();
	cout << "mode this->ndocs = " << this->ndocs << endl;
	this->nvocab = this->ptrndat->GetNumVocab();
	cout << "mode this->nvocab = " << this->nvocab << endl;

	//********* Initialize beta
	// initialize all beta to common word prior
	this->beta = new double*[this->nsentis];
	for (int s = 0; s < this->nsentis; ++s)
	{
		this->beta[s] = new double[this->nvocab];
		for (int v = 0; v < this->nvocab; ++v)
		{
			this->beta[s][v] = this->betas[0]; // initialize as common word
		}
	}

	// set sentiment words prior
	if (this->lexicon_dir != "") {
		// get unique sentiment word to set hyper-parameter beta
		vector<vector<int> > lexicon = this->ptrndat->GetLexicon();
		set<int> unique_senti_word;
		for (int s = 0; s < this->nsentis; ++s)
		{
			for (int v = 0; v < lexicon[s].size(); ++v)
			{
				unique_senti_word.insert(lexicon[s][v]);
			}
		}


		// set hyper-parameter of other sentiment words
		for (set<int>::iterator it = unique_senti_word.begin(); it != unique_senti_word.end(); ++it) {
			for (int s = 0; s < this->nsentis; ++s)
			{
				this->beta[s][*it] = this->betas[2];
			}
		}

		// set hyper-parameter of current sentiment words
		for (int s = 0; s < this->nsentis; ++s)
		{
			for (int v = 0; v < lexicon[s].size(); ++v)
			{
				this->beta[s][lexicon[s][v]] = this->betas[1];
			}
		}
	}


	this->sum_beta = new double[this->nsentis];
	for (int s = 0; s < this->nsentis; ++s)
	{
		this->sum_beta[s] = 0;
		for (int v = 0; v < this->nvocab; ++v)
		{
			this->sum_beta[s] += this->beta[s][v];
		}
	}


	this->theta = new double*[this->ndocs]; // D * T
	for (int d = 0; d < this->ndocs; ++d)
	{
		this->theta[d] = new double[this->ntopics];
		for (int t = 0; t < this->ntopics; ++t)
		{
			this->theta[d][t] = 0.0;
		}
	}


	this->pi = new double**[this->ndocs]; // D * T * S
	for (int d = 0; d < this->ndocs; ++d)
	{
		this->pi[d] = new double*[this->ntopics];
		for (int z = 0; z < this->ntopics; ++z)
		{
			this->pi[d][z] = new double[this->nsentis];
			for (int s = 0; s < this->nsentis; ++s)
			{
				this->pi[d][z][s] = 0.0;
			}
		}
	}

	this->phi = new double**[this->ntopics]; // T * S * V
	for (int t = 0; t < this->ntopics; ++t)
	{
		this->phi[t] = new double*[this->nsentis];
		for (int s = 0; s < this->nsentis; ++s)
		{
			this->phi[t][s] = new double[this->nvocab];
			for (int v = 0; v < this->nvocab; ++v)
			{
				this->phi[t][s][v] = 0.0;
			}
		}
	}

	this->doc_senti_prob = new double*[this->ndocs];
	for (int d = 0; d < this->ndocs; ++d)
	{
		this->doc_senti_prob[d] = new double[this->nsentis];
		for (int s = 0; s < this->nsentis; ++s)
		{
			this->doc_senti_prob[d][s] = 0.0;
		}
	}


	this->nDT = new int*[this->ndocs]; 
	for (int d = 0; d < this->ndocs; ++d)
	{
		this->nDT[d] = new int[this->ntopics];
		for (int t = 0; t < this->ntopics; ++t)
		{
			this->nDT[d][t] = 0;
		}
	}


	this->nDTS = new int**[this->ndocs];
	for (int d = 0; d < this->ndocs; ++d)
	{

		this->nDTS[d] = new int*[this->ntopics];
		for (int z = 0; z < this->ntopics; ++z)
		{
			this->nDTS[d][z] = new int[this->nsentis];
			for (int s = 0; s < this->nsentis; ++s)
			{
				this->nDTS[d][z][s] = 0;
			}
		}
	}


	this->nTSW = new int**[this->ntopics];
	for (int t = 0; t < this->ntopics; ++t)
	{
		this->nTSW[t] = new int*[this->nsentis];
		for (int s = 0; s < this->nsentis; ++s)
		{
			this->nTSW[t][s] = new int[this->nvocab];
			for (int v = 0; v < this->nvocab; ++v)
			{
				this->nTSW[t][s][v] = 0;
			}
		}
	}

	this->sumDT = new int[this->ndocs];
	for (int d = 0; d < this->ndocs; ++d)
	{
		this->sumDT[d] = 0;
	}

	// cout << "this->ndocs = " << this->ndocs << endl;
	// cout << "this->ntopics = " << this->ntopics << endl;
	this->sumDTS = new int*[this->ndocs];
	for (int d = 0; d < this->ndocs; ++d)
	{
		this->sumDTS[d] = new int[this->ntopics];
		for (int z = 0; z < this->ntopics; ++z)
		{
			this->sumDTS[d][z] = 0;
		}
	}

	this->sumTSW = new int*[this->ntopics];
	for (int t = 0; t < this->ntopics; ++t)
	{
		this->sumTSW[t] = new int[this->nsentis];
		for (int s = 0; s < this->nsentis; ++s)
		{
			this->sumTSW[t][s] = 0;
		}
	}

	cout << "Initialize topic and sentiment for estimating ..." << endl;
	if (this->lexicon_dir == "")
		randomize = true;


	srandom(time(0));  // random seed
	vector<vector<int> > senti_words = ptrndat->GetLexicon();
		
	for (auto pdoc: this->ptrndat->GetDocs()) {

		int doc_no = pdoc->GetDocNo();

		// cout << "doc_no" << doc_no << endl;

		for (auto psen : pdoc->GetSentences()) {

			// get random topic
			int new_topic = (int)(((double)random() / RAND_MAX) * this->ntopics);

			// cout << "new_topic = " << new_topic << endl;

			// set current topic
			psen->SetTopic(new_topic);

			// increase counters
			this->nDT[doc_no][new_topic] += 1;
			this->sumDT[doc_no] += 1;

			for (auto pseg : psen->GetSegments()) {

				// initailize sentiment assignment with lexicon prior knowledge
				int new_senti = -1;
				int nSentenceSenti = 0;
				
				for (auto pword : pseg->GetWords()) {
					int word_no = pword->GetWordNo();

					for (int si = 0; si < senti_words.size(); si++) {
						for (auto wi : senti_words[si]) {
							if (word_no == wi) {
								if (nSentenceSenti == 0 || si != new_senti) nSentenceSenti++;
								pword->SetLexicon(si);
								new_senti = si;
								break;
							}
						}
					}
				}
	
				pseg->SetNumSentenceSentis(nSentenceSenti);

				// If randomize or there is no sentiment words or 
				// more than one sentiment words, get random sentiment
				if (randomize || nSentenceSenti != 1 || new_senti == -1) {
					new_senti = (int)(((double)random() / RAND_MAX) * this->nsentis);
				}

				// 	set current sentiment
				// cout << "new_senti = " << new_senti << endl;
				pseg->SetSentiment(new_senti);

				map<int, int>::iterator it = psen->GetSentiCnt().find(new_senti);
				if (it != psen->GetSentiCnt().end()) {
					it->second += 1;
				} else {
					psen->GetSentiCnt().insert({new_senti, 1});
				}

				map<int, map<int, int> > senti_word_cnt = psen->GetSentiWordCnt();
				map<int, map<int, int> >::iterator its = senti_word_cnt.find(new_senti);
				map<int, int> word_cnt = pseg->GetWordCnt();
				if (its != senti_word_cnt.end()) {
					for (auto itw = word_cnt.begin(); itw != word_cnt.end(); itw++) {
						map<int, int>::iterator itsw = senti_word_cnt[new_senti].find(itw->first);
						if (itsw != senti_word_cnt[new_senti].end()) {
							itsw->second += 1;
						} else {
							senti_word_cnt[new_senti].insert({itw->first, 1});
						}
					}
				} else {
					senti_word_cnt.insert({new_senti, word_cnt});
				}

				// increase counters
				// cout << "increase counters" << endl;
				// cout << "increase nDTS: " << doc_no << " " << new_topic << " " << new_senti << endl;
				this->nDTS[doc_no][new_topic][new_senti] += 1;
				// cout << "increase sumDTS: " << doc_no << " " << new_topic << endl;
				this->sumDTS[doc_no][new_topic] += 1;

				// cout << "increase words" << endl;
				for (auto pword : pseg->GetWords()) {
					int word_no = pword->GetWordNo();
					this->nTSW[new_topic][new_senti][word_no] += 1;
					this->sumTSW[new_topic][new_senti] += 1;
				}

				// cout << "end segment" << endl;
			} // end segment
		}
	}

	return 0;
}


void Model::PrintConfig() {

	cout << "Model configuration:" << endl;
	cout << "\talpha = " << this->alpha << endl;
	cout << "\tgamma = " << this->gamma << endl;
	cout << "\tbeta = " << this->betas[0] << " " << this->betas[1] << " " << this->betas[2] << endl;


	cout << "\tdoc_dir = " << this->doc_dir << endl;
	if (this->lexicon_dir == "") 
		cout << "\tDon't use lexicon." << endl;
	else
		cout << "\tlexicon_dir = " << this->lexicon_dir << endl;
	cout << "\toutput_dir = " << this->output_dir << endl;

	cout << "\tntopics = " << this->ntopics << endl;
	cout << "\tnsentis = " << this->nsentis << endl;

	cout << "\tndocs = " << this->ndocs << endl;
	cout << "\tnvocab = " << this->nvocab << endl;
}


/**
 * Estimate topic and sentiment assignment for each sentences and segments
 * @param interval Interval to print iteration step.
 */
void Model::Estimate(int interval) {

	cout << "Estimating..." << endl;
	for (int iter = 0; iter < niters; ++iter)
	{
		if (iter % interval == 0 && iter >= interval) {
			cout << "Iteration #" << iter << endl;
		}

		for (auto pdoc: this->ptrndat->GetDocs()) {
			SamplingDoc(pdoc);
		}
	}

	cout << "Computing variables..." << endl;
	// calculate variables
	ComputeTheta();
	ComputePi();
	ComputePhi();
	ComputeDocSentiProb();

	cout << "Saving model..." << endl;
	// save model
	SaveModel();
}

/**
 * Estimate topic and sentiment assignment of a document.
 * 
 * The procedure is as follow:
 *  1. For each sentence in document:
 *  2.     sample a new topic
 *  3.     For each segment in this sentence with new topic:
 *  4.         sample new sentiment 
 *  	
 * @param pdoc pointer to a document
 */
void Model::SamplingDoc(Document* & pdoc) {

	int doc_no = pdoc->GetDocNo();
	for (auto psen : pdoc->GetSentences()) {

		SamplingSentence(psen, doc_no);

		int cur_topic = psen->GetTopic();
		for (auto pseg : psen->GetSegments()) {
			// If there is only one sentiment word in segment,
			// then the segment sentiment is set to the sentiment
			
			// if (pseg->GetNumSentenceSentis() == 1) continue;

			int old_senti = pseg->GetSentiment();
			SamplingSeg(pseg, doc_no, cur_topic);

			int new_senti = pseg->GetSentiment();
			if (new_senti == old_senti) continue;

			// update map in sentence
			psen->GetSentiCnt()[old_senti]--;
			psen->GetSentiCnt()[new_senti]++;

			map<int, int> word_cnt = psen->GetSentiWordCnt()[old_senti];
			for (auto it : pseg->GetWordCnt()) {
				word_cnt[it.first] -= it.second;
			}

			word_cnt = psen->GetSentiWordCnt()[new_senti];
			for (auto it : pseg->GetWordCnt()) {
				map<int, int>::iterator itw = word_cnt.find(it.first);
				if (itw != word_cnt.end()) {
					itw->second += it.second;
				} else {
					word_cnt[it.first] = it.second;
				}
			}
		}
	}
}



void Model::SamplingSentence(Sentence* & psen, int doc_no) {

	int cur_topic = psen->GetTopic();

	this->nDT[doc_no][cur_topic]--;
	this->sumDT[doc_no]--;

	for (auto pseg : psen->GetSegments()) {
		int cur_senti = pseg->GetSentiment();

		this->nDTS[doc_no][cur_topic][cur_senti]--;
		this->sumDTS[doc_no][cur_topic]--;

		for (auto pword : pseg->GetWords()) {
			int word_no = pword->GetWordNo();
			this->nTSW[cur_topic][cur_senti][word_no]--;
			this->sumTSW[cur_topic][cur_senti]--;
		}
	}

	map<int, int> senti_cnt = psen->GetSentiCnt();
	map<int, map<int, int> > senti_word_cnt = psen->GetSentiWordCnt();

	double *prob_topic = new double[this->ntopics];
	for (int ti = 0; ti < this->ntopics; ti++) {

		double gamma0 = this->sumDTS[doc_no][ti] + this->sum_gamma;
		int n0 = 0;
		double expectDTS = 1.0;

		for (auto it = senti_cnt.begin(); it != senti_cnt.end(); it++) {
			int si = it->first;
			int cnt = it->second;
			double gammal = this->nDTS[doc_no][ti][si] + this->gamma;

			for (int ns = 0; ns < cnt; ns++) {
				expectDTS *= (gammal + ns) / (gamma0 + n0);
				n0++;
			}

			map<int, int> word_cnt = senti_word_cnt[si];
			double beta0 = this->sumTSW[ti][si] + this->sum_beta[si];
			int nw0 = 0;
			double expectTSW = 1.0;

			for (auto itw = word_cnt.begin(); itw != word_cnt.end(); itw++) {
				int word_no = itw->first;
				int wcnt = itw->second;
				double betaw = this->nTSW[ti][si][word_no] + this->beta[si][word_no];

				for (int nw = 0; nw < wcnt; nw++) {
					expectTSW *= (betaw + nw) / (beta0 + nw0);
					nw0++;
				}
			}
			expectDTS *= expectTSW;

		}

		prob_topic[ti] = (this->nDT[doc_no][ti] + this->alpha) * expectDTS;
	}

	int new_topic = SampleNew(prob_topic, this->ntopics);
	this->nDT[doc_no][new_topic]++;
	this->sumDT[doc_no]++;

	for (auto pseg : psen->GetSegments()) {
		int cur_senti = pseg->GetSentiment();

		this->nDTS[doc_no][new_topic][cur_senti]++;
		this->sumDTS[doc_no][new_topic]++;

		for (auto pword : pseg->GetWords()) {
			int word_no = pword->GetWordNo();
			this->nTSW[new_topic][cur_senti][word_no]++;
			this->sumTSW[new_topic][cur_senti]++;
		}
	}

}

void Model::SamplingSeg(Segment* & pseg, int doc_no, int cur_topic) {

	int cur_senti = pseg->GetSentiment();
	this->nDTS[doc_no][cur_topic][cur_senti]--;
	this->sumDTS[doc_no][cur_topic]--;

	for (auto pword : pseg->GetWords()) {
		int word_no = pword->GetWordNo();
		this->nTSW[cur_topic][cur_senti][word_no]--;
		this->sumTSW[cur_topic][cur_senti]--;
	}


	map<int, int> word_cnt = pseg->GetWordCnt();
	double *prob_senti = new double[this->nsentis];
	for (int si = 0; si < this->nsentis; si++) {
		bool trim = false;

		// fast trim: if there is a lexicon word,
		// and its lexicon is not si, then there is segment cannot
		// be assigned as si, thus no need to calculate it
		for (auto pword : pseg->GetWords()) {
			int lexicon = pword->GetLexicon();
			if (lexicon != -1 && lexicon != si) {
				trim = true;
				break;
			}
		}

		if (trim) {
			prob_senti[si] = 0;
		} else {
			double beta0 = this->sumTSW[cur_topic][si] + this->sum_beta[si];
			double nw0 = 0;
			double expectTSW = 1.0;

			for (auto it = word_cnt.begin(); it != word_cnt.end(); it++) {
				int word_no = it->first;
				// int word_no = pword->GetWordNo();
				int cnt = it->second;

				double betaw = this->nTSW[cur_topic][si][word_no] + this->beta[si][word_no];
				for (int nw = 0; nw < cnt; nw++) {
					expectTSW *= (betaw + nw) / (beta0 + nw0);
					nw0++;
				}
			}
			prob_senti[si] = (this->nDTS[doc_no][cur_topic][si] + this->gamma) / (this->sumDTS[doc_no][cur_topic])
			* expectTSW;
		}
	}

	int new_senti = SampleNew(prob_senti, this->nsentis);
	pseg->SetSentiment(new_senti);

	this->nDTS[doc_no][cur_topic][new_senti]++;
	this->sumDTS[doc_no][cur_topic]++;

	for (auto pword : pseg->GetWords()) {
		int word_no = pword->GetWordNo();
		this->nTSW[cur_topic][new_senti][word_no]++;
		this->sumTSW[cur_topic][new_senti]++;
	}	
}



/**
 * Sample a random int number between 0 and length-1 with probability distribution.
 * @param  prob   probability distribution
 * @param  length length of probs
 * @return        the random number
 */
int Model::SampleNew(double * &prob, int length) {
	for (int i = 1; i < length; ++i)
	{
		prob[i] += prob[i-1];
	}

	double u = ((double)random() / RAND_MAX) * prob[length - 1];
	for (int i = 0; i < length; ++i)
	{
		if (prob[i] > u)
			return i;
	}
	return length-1;
}


void Model::ComputeTheta() {
	for (int d = 0; d < this->ndocs; ++d)
	{
		for (int t = 0; t < this->ntopics; ++t)
		{
			this->theta[d][t] = (this->nDT[d][t] + this->alpha) 
								/ (this->sumDT[d] + this->sum_alpha);
		}
	}
}

void Model::ComputePi() {
	for (int d = 0; d < this->ndocs; ++d)
	{
		for (int t = 0; t < this->ntopics; ++t) 
		{
			for (int s = 0; s < this->nsentis; ++s)
			{
				this->pi[d][t][s] = (this->nDTS[d][t][s] + this->gamma) 
									/ (this->sumDTS[d][t] + this->sum_gamma);
			}
		}
		
	}
}

void Model::ComputePhi() {
	for (int t = 0; t < this->ntopics; ++t)
	{
		for (int s = 0; s < this->nsentis; ++s)
		{
			for (int v = 0; v < this->nvocab; ++v)
			{
				this->phi[t][s][v] = (this->nTSW[t][s][v] + this->beta[s][v]) / 
									(this->sumTSW[t][s] + this->sum_beta[s]);
			}
		}
	}
}

void Model::ComputeDocSentiProb() {
	for (int d = 0; d < this->ndocs; ++d)
	{
		double sum = 0.0;
		for (int s = 0; s < this->nsentis; ++s)
		{
			double prob = 0.0;
			for (int t = 0; t < this->ntopics; ++t)
			{
				prob += this->nDT[d][t] * this->nDTS[d][t][s];
			}
			this->doc_senti_prob[d][s] = prob;
			sum += prob;
		}

		for (int s = 0; s < this->nsentis; ++s)
		{
			this->doc_senti_prob[d][s] /= sum;
		}
	}
}


int Model::SaveModel() {

	string model_name = to_string(this->ttimes) + "_est_";
	model_name += "ntopics" + to_string(this->ntopics) + "_";
	model_name += "nsentis" + to_string(this->nsentis);
	if (this->lexicon_dir == "") 
		model_name += "(0)_";
	else
		model_name += "(" + to_string(this->nsentis) + ")_";
	model_name += "niters" + to_string(this->niters) + "_";
	model_name += "alpha" + to_string(this->alpha) + "_";
	model_name += "gamma" + to_string(this->gamma) + "_";
	model_name += "betas" + to_string(this->betas[0]) 
	+ to_string(this->betas[1]) + to_string(this->betas[2]);

	cout << "model_name = " << model_name << endl;

	if (SaveModelTheta(model_name)) {
		cout << "Error in save theta!" << endl;
		return 1;
	}

	if (SaveModelPi(model_name)) {
		cout << "Error in save pi!"  << endl;
		return 1;
	}

	if (SaveModelPhi(model_name)) {
		cout << "Error in SaveModelPhi" << endl;
		return 1;
	}

	
	if (SaveModelDocSentiProb(model_name)) {
		cout << "Error in SaveModelDocSentiProb" << endl;
		return 1;
	}


	if (ptrndat->ReadVocab(this->doc_dir, true)) {
		cout << "Error in ReadVocab reverse!" << endl;
		return 1;
	}

	if (SaveModelTopWords(model_name)) {
		cout << "Error in save top words!" << endl;
		return 1;
	}

	if (SaveModelAssign(model_name)) {
		cout << "Error in save assignment!" << endl;
		return 1;
	}

	return 0;
}


int Model::SaveModelTheta(string model_name) {

	cout << "SaveModelTheta" << endl;

	string file_name = this->output_dir + "/theta_" + model_name + ".txt";
	ofstream ofile(file_name);
	if (ofile.is_open()) {
		for (int d = 0; d < this->ndocs; ++d)
		{
			// ofile << std::fixed << std::setprecision(4) << this->theta[d][t] << " ";

			for (int t = 0; t < this->ntopics; ++t)
			{
				ofile << std::fixed << std::setprecision(4) << this->theta[d][t] << " ";
			}
			ofile << "\n";
		}
		ofile.close();
	} else {
		cout << "Error open file " << file_name << endl;
		return 1;
	}
	return 0;
}

int Model::SaveModelPi(string model_name) {

	cout << "SaveModelPi" << endl;

	string file_name = this->output_dir + "/pi_" + model_name + ".txt";
	ofstream ofile(file_name);
	if (ofile.is_open()) {
		for (int d = 0; d < this->ndocs; ++d)
		{
			for (int t = 0; t < this->ntopics; ++t)
			{
				for (int s = 0; s < this->nsentis; ++s)
			{
				ofile << std::fixed << std::setprecision(4) << this->pi[d][t][s] << " ";
			}
			ofile << "\n";	
			}
			ofile << "\n";
		}
		ofile.close();
	} else {
		cout << "Error open file_name" << file_name << endl;
		return 1;
	}
	return 0;
}

int Model::SaveModelPhi(string model_name) {

	cout << "SaveModelPhi" << endl;

	string file_name = this->output_dir + "/phi_" + model_name + ".txt";
	ofstream ofile(file_name);
	if (ofile.is_open()) {
		for (int t = 0; t < this->ntopics; ++t)
		{
			for (int s = 0; s < this->nsentis; ++s)
			{
				for (int v = 0; v < this->nvocab; ++v)
				{
					ofile << std::fixed << std::setprecision(4) << this->phi[t][s][v] << " ";
				}
				ofile << "\n";
			}
			ofile << "\n";
		}
		ofile.close();
	} else {
		cout << "Error open file_name" << file_name << endl;
		return 1;
	}
	return 0;
}


int Model::SaveModelDocSentiProb(string model_name) {

	cout << "SaveModelDocSentiProb" << endl;

	string file_name = this->output_dir + "/doc_senti_prob_" + model_name + ".txt";
	ofstream ofile(file_name);
	if (ofile.is_open()) {

		for (int d = 0; d < this->ndocs; ++d)
		{
			for (int s = 0; s < this->nsentis; ++s)
			{
				ofile << std::fixed << std::setprecision(4) << this->doc_senti_prob[d][s] << " ";
			}
			ofile << "\n";
		}

		ofile.close();
	}else {
		cout << "Error open file_name" << file_name << endl;
		return 1;
	}
	return 0;
}


int Model::SaveModelTopWords(string model_name, int topk) {

	cout << "SaveModelTopWords" << endl;


	if (topk > this->nvocab)
		topk = this->nvocab;

	map<int, string> id2word = this->ptrndat->GetId2Word();
	map<int, string>::iterator it;

	string file_name = this->output_dir + "/top" + to_string(topk) + "_words_" + model_name + ".txt";	
	ofstream ofile(file_name);
	if (ofile.is_open()) {
		for (int t = 0; t < this->ntopics; ++t)
		{
			for (int s = 0; s < this->nsentis; ++s)
			{
				
				vector<pair<int, double> > word_probs;
				pair<int, double> word_prob;
				for (int v = 0; v < this->nvocab; ++v)
				{
					word_prob.first = v;
					word_prob.second = this->phi[t][s][v];
					word_probs.push_back(word_prob);
				}

				Utils::QuickSort(word_probs, 0, word_probs.size() - 1);

				ofile << "Topic #" << t << " Sentiment #" << s << "\n";
				for (int k = 0; k < topk; ++k)
				{
					it = id2word.find(word_probs[k].first);
					if (it != id2word.end()) {
						ofile << "\t" << (it->second).c_str() << " " << word_probs[k].second << "\n";
					}
				}
			}
		}
		ofile.close();
	} else {
		cout << "Error open file_name" << file_name << endl;
		return 1;
	}
	return 0;
}


int Model::SaveModelAssign(string model_name) {

	cout << "SaveModelAssign" << endl;


	map<int, string> id2word = this->ptrndat->GetId2Word();

	string file_name = this->output_dir + "/assignment_" + model_name + ".txt";	
	ofstream ofile(file_name);
	if (ofile.is_open()) {

		for (auto pdoc: this->ptrndat->GetDocs()) {

				ofile << pdoc->GetDocId() << "\t";

			for (auto psen : pdoc->GetSentences()) {

				int cur_topic = psen->GetTopic();
				ofile << "t" << to_string(cur_topic) << ":{";

				for (auto pseg : psen->GetSegments()) {

					int cur_senti = pseg->GetSentiment();
					ofile << "s" << to_string(cur_senti) << ":{";

					for (auto pword : pseg->GetWords()) {
						int word_no = pword->GetWordNo();
						ofile << id2word[word_no] << ",";
						// ofile << to_string(word_no) << ",";
					}
					ofile << "}\t";
				}
				ofile << "}\t\t";
			}
			ofile << '\n';
		}
		ofile.close();
	} else {
		cout << "Error open file_name" << file_name << endl;
		return 1;
	}
	return 0;
}

