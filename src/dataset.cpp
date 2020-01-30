/*
 * Copyright (C) 2017 by
 * 
 * 	Yang Qinjuan
 *	yangqinj@mail2.sysu.edu.cn
 * 	Computer and Data Science
 * 	Sun Yat-sat University
 *
 */

#include "dataset.h"
#include <fstream>
#include <iostream>
#include <string>
#include "utils.h"

using std::cout;
using std::ifstream;
using std::endl;
using std::shared_ptr;
using std::make_shared;
using std::move;

// ************************************************
// 			Word 
// ************************************************
Word::Word(){
	this->lexicon = -1;
	this->word_no = -1;
}

Word::Word(int word_no) {
	this->lexicon = -1;
	this->word_no = word_no;
}

Word::Word(int word_no, int lexicon) {
	this->word_no = word_no;
	this->lexicon = lexicon;
}

Word::~Word() {}

void Word::SetWordNo(int word_no) {
	this->word_no = word_no;
}

void Word::SetLexicon(int lexicon) {
	this->lexicon = lexicon;
}

int Word::GetLexicon() {
	return this->lexicon;
}

int Word::GetWordNo() {
	return this->word_no;
}

// ************************************************
// 			Segment 
// ************************************************
Segment::Segment() {
	this->sentiment = -1;
}

Segment::~Segment() {}

void Segment::SetSentiment(int sentiment) {
	this->sentiment = sentiment;
}

void Segment::SetWords(vector<shared_ptr<Word>> &words) {
	this->words = words;

	for (auto pword : words) {
		int word_no = pword->GetWordNo();
		map<int, int>::iterator it = this->word_cnt.find(word_no);
		if (it != this->word_cnt.end()) {
			it->second += 1;
		} else {
			this->word_cnt[word_no] = 1;
		}
	}
}

int Segment::GetSentiment() {
	return this->sentiment;
}

const vector<shared_ptr<Word>>& Segment::GetWords() const {
	return this->words;
}

const map<int, int>& Segment::GetWordCnt() const {
	return this->word_cnt;
}

void Segment::SetNumSentenceSentis(int n) {
	this->num_sentence_sentis = n;
}

int Segment::GetNumSentenceSentis() {
	return this->num_sentence_sentis;
}

// ************************************************
// 			Sentence 
// ************************************************
Sentence::Sentence(int nsentis){
	this->topic = -1;

	this->senti_cnt = new int[nsentis];
	for (int i = 0; i < nsentis; i++) {
		this->senti_cnt[i] = 0;
	}

	for (int i = 0; i < nsentis; i++) {
		this->senti_word_cnt.push_back(map<int, int>());
	}
}

Sentence::~Sentence() {
	if (this->senti_cnt) {
		delete this->senti_cnt;
	}
}

void Sentence::SetTopic(int topic) {
	this->topic = topic;
}

void Sentence::SetSegment(vector<shared_ptr<Segment>> &segments) {
	this->segments = segments;
}

int Sentence::GetTopic() {
	return this->topic;
}

const vector<shared_ptr<Segment>>& Sentence::GetSegments() const {
	return this->segments;
}

const int* Sentence::GetSentiCnt() const {
	return this->senti_cnt;
}

const vector<map<int, int>>& Sentence::GetSentiWordCnt() const {
	return this->senti_word_cnt;
}

void Sentence::IncreaseSentiCnt(int senti, int cnt) {
	this->senti_cnt += cnt;
}

void Sentence::DecreaseSentiCnt(int senti, int cnt) {
	this->senti_cnt -= cnt;
}

void Sentence::IncreaseSentiWordCnt(int senti, int word, int cnt) {
	auto it = senti_word_cnt[senti].find(word);
	if (it != senti_word_cnt[senti].end()) {
		it->second += cnt;
	} else {
		senti_word_cnt[senti].insert({word, cnt});
	}
}

void Sentence::DecreaseSentiWordCnt(int senti, int word, int cnt) {
	auto it = senti_word_cnt[senti].find(word);
	if (it != senti_word_cnt[senti].end()) {
		it->second -= cnt;
	}
}

// ************************************************
// 			Document 
// ************************************************
Document::Document() {
	this->doc_no = -1;
}

Document::Document(int doc_no) {
	this->doc_no = doc_no;
}

Document::~Document() {}

void Document::SetDocNo(int doc_no) {
	this->doc_no = doc_no;
}

void Document::SetDocId(string doc_id) {
	this->docId = doc_id;
}

void Document::SetSentences(vector<shared_ptr<Sentence>> &sentences) {
	this->sentences = sentences;
}

int Document::GetDocNo() {
	return this->doc_no;
}

string Document::GetDocId() {
	return this->docId;
}

const vector<shared_ptr<Sentence>>& Document::GetSentences() const {
	return this->sentences;
}

// ************* Dataset ********
Dataset::Dataset() {
	this->ndocs = 0;
	this->nvocab = 0;
	this->nsentis = 0;
}

Dataset::~Dataset() {}

void Dataset::SetNumSentis(int nsentis) {
	this->nsentis = nsentis;
}

/**
 * Read docuemnt ids from file
 * @param  doc_dir directory of file
 * @return         whether success.
 */
int Dataset::ReadDocs(string doc_file) {
	cout << "Reading documents:" << doc_file << " ..." << endl;

	ifstream ifile(doc_file);

	if (ifile.is_open()) {
		int doc_no = 0;
		string doc_str, line, doc_id;

		while (getline(ifile, line)) { // read a document ids string
			vector<string> terms = Utils::Split(line, ":");
			doc_id = terms[0];
			doc_str = terms[1];

			shared_ptr<Document> pdoc = make_shared<Document>();
			pdoc->SetDocNo(doc_no++);
			pdoc->SetDocId(doc_id);

			vector<string> sentences_str = Utils::Split(doc_str, "\t\t");
			vector<shared_ptr<Sentence>> sentences;

			for (auto sen : sentences_str) { // for each sentence ids string
				shared_ptr<Sentence> psentence = make_shared<Sentence>(this->nsentis);
				vector<string> segs_str = Utils::Split(sen, "\t");
				vector<shared_ptr<Segment>> segments;

				for (auto seg : segs_str) { // for each segment ids string
					shared_ptr<Segment> psegment = make_shared<Segment>();
					vector<string> words_str = Utils::Split(seg, " ");
					vector<shared_ptr<Word>> words; // word ids int

					for (auto word_str : words_str) { // for each word ids string
						int word_id = stoi(word_str);
						shared_ptr<Word> pword = make_shared<Word>(word_id);
						words.emplace_back(move(pword));
					}
					psegment->SetWords(words);
					segments.emplace_back(move(psegment));
				}
				psentence->SetSegment(segments);
				sentences.emplace_back(move(psentence));
			}
			pdoc->SetSentences(sentences);
			this->docs.emplace_back(move(pdoc));
		}
	}
	this->ndocs = this->docs.size();
	if (this->ndocs == 0) {
		cout << "Empty documents!" << endl;
		return 1;
	}
	cout << "numer of documents is " << this->ndocs << endl;
	return 0;
}

/**
 * Read sentiment lexicon words.
 * @param  lexicon_dir directory of the lexicon file. If directory is empty, 
 *                     it means doesn't use the lexicon and the prior of all words are same.
 * @return             whether read success.
 */
int Dataset::ReadLexicon(string lexicon_dir) {
	if (lexicon_dir == "") {
		cout << "Does not use lexicon prior!" << endl;
		return 0;
	}

	cout << "Reading lexicon ..." << endl;
	int total_senti_words = 0;
	string lexiocn_preffix = "SentiWords_";

	for (int s = 0; s < this->nsentis; ++s) {
		int senti_words_cnt = 0;
		string lexicon_file = lexicon_dir + "/" + lexiocn_preffix 
		+ to_string(s) + ".txt";

		vector<int> senti_words;
		ifstream ifile(lexicon_file);

		if (ifile.is_open()) {
			string word;
			while (getline(ifile, word)) {
				word.pop_back();
				map<string,int>::iterator it = this->word2id.find(word);
				if (it != this->word2id.end()) {
					senti_words.push_back(it->second);
					senti_words_cnt += 1;
				}
			}
			ifile.close();
		}

		if (senti_words_cnt == 0) {
			cout << "Empty sentiment words in sentiment " << s << "!" << endl;
			return 1;
		}

		this->lexicon_words.emplace_back(std::move(senti_words));
		cout << "number of words in sentiment " << s << " is: " 
		     << senti_words_cnt << endl;
		total_senti_words += senti_words_cnt;
	}

	cout << "total number of words in lexicon is: " 
       << total_senti_words << endl;

	return 0;
}

/**
 * Read vocabulary from file. 
 * @param  vocab_dir directory of file
 * @param  reverse   If true, read word2id <word, id>. Otherwise, read id2word <int, word>
 * @return           [description]
 */
int Dataset::ReadVocab(string vocab_dir, bool reverse) {
	string vocab_file = vocab_dir + "/" + "vocabulary.txt";
	ifstream ifile(vocab_file);

	string word;
	int id = 1; // Word Id starts from 1.
	if (!reverse) { // read <word, id>
		while (getline(ifile, word)) {
			if (word != "") {
				// word.pop_back(); // pop back '\n'
				this->word2id[word] = id;
				id++;
			}
		}
		this->nvocab = this->word2id.size();
	} else {
		while (getline(ifile, word)) {
			if (word != "") {
				// word.pop_back(); // pop back '\n'
				this->id2word[id] = word;  
				id++;
			}
		}
		this->nvocab = this->id2word.size();
	}
	
	if (this->nvocab == 0) {
		cout << "Empty vocabulary!" << endl;
		return 1;
	}
	cout << "vocabulary size is " << this->nvocab << endl;
	return 0;
}

const vector<shared_ptr<Document>>& Dataset::GetDocs() const {
	return this->docs;
}

const vector<vector<int>>& Dataset::GetLexicon() const {
	return this->lexicon_words;
}

int Dataset::GetNumDocs() {
	return this->ndocs; 
}

int Dataset::GetNumVocab() { 
	return this->nvocab; 
}

const map<int, string>& Dataset::GetId2Word() const {
	return this->id2word;
}