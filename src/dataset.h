/*
 * Copyright (C) 2017 by
 * 
 * 	Yang Qinjuan
 *	yangqinj@mail2.sysu.edu.cn
 * 	Computer and Data Science
 * 	Sun Yat-sat University
 *
 */


#ifndef _DATASET_H_
#define _DATASET_H_ 

#include <vector>
#include <map>
#include <string>

using namespace std;

class Word
{
private:
	int word_no; // No. of word

	int lexicon; // If it is a sentiment word, lexicon is set to 
				// the corresponding sentiment. Otherwise, lexicon is -1.

public:
	Word();
	Word(int);
	Word(int, int);
	~Word();
	void SetWordNo(int word_no);
	void SetLexicon(int);
	int GetLexicon();
	int GetWordNo();
	
};


class Segment
{
private:
	int sentiment;
	vector<Word*> words;
	map<int, int> word_cnt; // key is word and
							  // value is the number of this words in 
							  // this segment
	int num_sentence_sentis;

public:
	Segment();
	~Segment();
	void SetSentiment(int);
	void SetWords(vector<Word*> &);
	int GetSentiment();
	vector<Word*> GetWords();
	map<int, int> GetWordCnt();
	void SetNumSentenceSentis(int);
	int GetNumSentenceSentis();
};


class Sentence
{
private:
	int topic;
	vector<Segment*> segments;
	
	map<int, int> senti_cnt; // key is the sentiment polarity, 
							// value is the  number of segments assignment 
							// to this sentiment
	// int* senti_cnt;
	map<int, map<int, int> > senti_word_cnt; // key is the sentiment polarity, 
											 // value is a map, whose key is word and
											 // value is the number of this words in 
											 // this sentence
	// map<int, map<int, int> > senti_word_cnt;

public:
	Sentence();
	~Sentence();
	void SetTopic(int topic);
	void SetSegment(vector<Segment*> &segments);
	int GetTopic();
	vector<Segment*> GetSegments();
	map<int, int> GetSentiCnt();
	map<int, map<int, int> > GetSentiWordCnt();
};


class Document
{
private:
	int doc_no; // No. of document in program 
	string docId;  // Id of document in dataset
	vector<Sentence*> sentences;
public:
	Document();
	Document(int);
	~Document();
	void SetDocNo(int doc_no);
	void SetDocId(string doc_id);
	void SetSentences(vector<Sentence*> &sentences);
	int GetDocNo();
	string GetDocId();
	vector<Sentence*> GetSentences();
};

/**
 * Class of the dataset, which cotains the documents, vocabulary 
 * and the sentiment lexicon.
 */
class Dataset
{
private:
	vector<Document*> docs;
	int ndocs; // number of documents

	map<string, int> word2id; // key is word string, value is its id
	map<int, string> id2word; // key is word id, value is its string
	int nvocab; // vocabulary size

	int nsentis; // number of sentiments setting by the model, 
				 // not the actually used sentiments in lexicon.
	vector<vector<int> >  lexicon_words; // sentiment lexicon of word ids

public:
	Dataset();
	~Dataset();
	void SetNumSentis(int);
	int ReadDocs(string);
	int ReadLexicon(string);
	int ReadVocab(string, bool reverse=false);
	vector<Document*> GetDocs();
	vector<vector<int> > GetLexicon();
	int GetNumDocs();
	int GetNumVocab();
	map<int, string> GetId2Word();
	// map<int, int> GetId2Lexicon();
	
};


#endif
