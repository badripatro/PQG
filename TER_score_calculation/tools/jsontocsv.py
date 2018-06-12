import numpy as np
#import pandas as pd
import cv2
import os
import glob
import math
import json

import sys

f = open('csv_files/resuts_json.csv', 'w')
val_result= json.load(open('check_point_json/resuts_json.json', 'r'))
print len(val_result)
for i in range(0,len(val_result)):
	if str(val_result[i]['question'].encode('utf-8'))== "":  ## this is to replace nil value with some garbage. other wise error in calucate ter score 
		f.write(str('W'))
	f.write(str(val_result[i]['question'].encode('utf-8'))+'\n')	# f.write(value+'\n')
f.close()


f = open('csv_files/quora_prepro_test_updated_int.csv', 'w')
val= json.load(open('quora_prepro_test_updated_int.json', 'r'))
val=val['questions']
print len(val)
for i in range(0,len(val_result)):
	f.write(str(val[i]['question'].encode('utf-8'))+'\n')	# f.write(value+'\n')
f.close()

