import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--output', type=str, required=True, help='ouput_file_name')
args = parser.parse_args()

root_path='/media/data/CCF_data/car_forum_data'

test_file=os.path.join(root_path,'test.csv')
test_data=pd.read_csv(test_file)
predict_file='predict_results.txt'
pred_data=pd.read_csv(predict_file,sep='\t')
# pred_data.head()
merge=pd.concat([test_data,pred_data],axis=1)
filename = args.output
# output_path=os.path.join('/media/data/CCF_data','submission_bert.csv')
merge.to_csv("submission_"+filename+".csv",columns=['id','label'],index=False)