import subprocess
import os

FNULL = open(os.devnull, 'w')    #use this if you want to suppress output to stdout from the subprocess

dataset = 'TanHotel'

lexicon_names = ["no_lexicon"]
for lexicon_name in lexicon_names:

	arg_lexicon = ""
	if lexicon_name != 'no_lexicon':
		arg_lexicon = "-lexicon_dir ./lexicon/" + lexicon_name + " "

	for ntopics in range(100,101,1):
		for ttimes in range(1,2,1):
			arg_ttime = "-ttimes " + str(ttimes) + " "
			arg_topic = "-ntopics " + str(ntopics) + " " 
			arg_iter = "-niters 1 " # number of iterations
			arg_senti = "-nsentis 2 " #  number of sentiments for estimating
			arg_doc = "-doc_dir ./dataset/" + dataset + " "

			arg_output = "-output_dir ./dataset/" + dataset + "/" + lexicon_name + "/" + str(ttimes)

			args = "./src/main " + arg_ttime + arg_topic + arg_iter + arg_senti + arg_doc + arg_lexicon + arg_output

			os.system(args)
