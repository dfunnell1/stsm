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

#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::to_string;
using std::vector;
using std::shared_ptr;

class Word {
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

class Segment {
 private:
	int sentiment;
	vector<shared_ptr<Word>> words;
	map<int, int> word_cnt; // key is word and
											    // value is the number of this words in 
											    // this segment
	int num_sentence_sentis;

 public:
	Segment();
	~Segment();
	void SetSentiment(int);
	void SetWords(vector<shared_ptr<Word>> &words);
	int GetSentiment();
	const vector<shared_ptr<Word>>& GetWords() const;
	const map<int, int>& GetWordCnt() const;
	void SetNumSentenceSentis(int);
	int GetNumSentenceSentis();
};

class Sentence {
 private:
	int topic;
	vector<shared_ptr<Segment>> segments;
	int* senti_cnt; // number of segments assigned to different sentiment polarities
	vector<map<int, int>> senti_word_cnt; // number of words assigned to 
																			  //different sentiment polarities
 public:
	Sentence(int nse);
	~Sentence();
	void SetTopic(int topic);
	void SetSegment(vector<shared_ptr<Segment>> &segments);
	int GetTopic();
	const vector<shared_ptr<Segment>>& GetSegments() const;
	const int* GetSentiCnt() const;
	const vector<map<int, int>> &GetSentiWordCnt() const;
	void IncreaseSentiCnt(int senti, int cnt = 1);
	void DecreaseSentiCnt(int senti, int cnt = 1);
	void IncreaseSentiWordCnt(int senti, int word_id, int cnt = 1);
	void DecreaseSentiWordCnt(int senti, int word_id, int cnt = 1);
};

class Document {
 private:
	int doc_no; // No. of document in program 
	string docId; // Id of document in dataset
	vector<shared_ptr<Sentence>> sentences;
 public:
	Document();
	Document(int);
	~Document();
	void SetDocNo(int doc_no);
	void SetDocId(string doc_id);
	void SetSentences(vector<shared_ptr<Sentence>> &sentences);
	int GetDocNo();
	string GetDocId();
	const vector<shared_ptr<Sentence>>& GetSentences() const;
};

class Dataset {
 private:
	vector<shared_ptr<Document>> docs;
	int ndocs; // number of documents

	map<string, int> word2id; // key is word string, value is its id
	map<int, string> id2word; // key is word id, value is its string
	int nvocab; // vocabulary size

	int nsentis; // number of sentiments setting by the model, 
				       // not the actually used sentiments in lexicon.
	vector<vector<int>> lexicon_words; // sentiment lexicon of word ids

 public:
	Dataset();
	~Dataset();
	void SetNumSentis(int);
	int ReadDocs(string);
	int ReadLexicon(string);
	int ReadVocab(string, bool reverse=false);
	const vector<shared_ptr<Document>>& GetDocs() const;
	const vector<vector<int>>& GetLexicon() const;
	int GetNumDocs();
	int GetNumVocab();
	const map<int, string>& GetId2Word() const;
};

#endif
