import os
import json
val= json.load(open('quora_prepro_test.json', 'r'))
out=[]
outermost = {}
for i in range(0,len(val)):

	# imgid = str(val[i]['ques_id'])[:-1]
	#     qcapid = val[i]['ques_cap_id']
	# quesid = val[i]['question_index']
	# ques = val[i]['question']
	# imgpath = val[i]['image_filename']
	# img_id = val[i]['image_index']

	jimg = {}
	#jimg['caption'] = capt
	jimg['question'] = val[i]['question']
	jimg['question1'] = val[i]['question1']
	# jimg['img_path'] = imgpath
	jimg['ques_id'] =  val[i]['ques_id']
	jimg['ques_id1'] =  val[i]['ques_id1']
	# jimg['para_id'] =  paraid
	jimg['para_id'] =  int(val[i]['id'])
	##########################################################
	out.append(jimg)

outermost['questions'] = out

print len(out)
json.dump(outermost, open('quora_prepro_test_updated_int.json', 'w'))