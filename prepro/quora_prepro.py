import csv
import json
import os

def main():
	out = []
	outtest = []
	outval = []
	with open('../data/quora_duplicate_questions.tsv','rb') as tsvin:
		tsvin = csv.reader(tsvin, delimiter=',')#read the tsv file of quora question pairs
		count0 = 1
		count1 = 1
		counter = 1
		for row in tsvin:
			counter = counter+1
			if row[5]=='0' and row[4][-1:]=='?':#the 6th entry in every row has value 0 or 1 and it represents paraphrases if that value is 1
				count0=count0+1
			elif row[5]=='1' and row[4][-1:]=='?':
				count1=count1+1
				if count1>1 and count1<100002:#taking the starting 1 lakh pairs as train set. Change this to 50002 for taking staring 50 k examples as train set
					# get the question and unique id from the tsv file
					quesid = row[1] #first question id
					ques = row[3] #first question
					img_id = row[0] #unique id for every pair
					ques1 = row[4]#paraphrase question
					quesid1 =row[2]#paraphrase question id				
					
					# set the parameters of json file for writing 
					jimg = {}

					jimg['question'] = ques
					jimg['question1'] = ques1
					jimg['ques_id'] =  quesid
					jimg['ques_id1'] =  quesid1
					jimg['id'] =  img_id

					out.append(jimg)

				elif count1>100001 and count1<130002:#next 30k as the test set acc to https://arxiv.org/pdf/1711.00279.pdf
					quesid = row[1] 
					ques = row[3] 
					img_id = row[0] 
					ques1 = row[4]
					quesid1 =row[2]

					jimg = {}

					jimg['question'] = ques
					jimg['question1'] = ques1
					jimg['ques_id'] =  quesid
					jimg['ques_id1'] =  quesid1
					jimg['id'] =  img_id

					outtest.append(jimg)	
				else :#rest as val
					quesid = row[1] 
					ques = row[3] 
					img_id = row[0] 
					ques1 = row[4]
					quesid1 =row[2]
				
					jimg = {}
					jimg['question'] = ques
					jimg['question1'] = ques1
					jimg['ques_id'] =  quesid
					jimg['ques_id1'] =  quesid1
					jimg['id'] =  img_id
					
					outval.append(jimg)
	#write the json files for train test and val
	print len(out)
	json.dump(out, open('../data/quora_raw_train.json', 'w'))
	print len(outtest)
	json.dump(outtest, open('../data/quora_raw_test.json', 'w'))
	print len(outval)
	json.dump(outval, open('../data/quora_raw_val.json', 'w'))


if __name__ == "__main__":
	main()