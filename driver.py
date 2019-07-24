#!/usr/bin/env python

from sklearn.externals import joblib
import numpy as np, os, os.path, sys, argparse
#from collections import defaultdict
from get_sepsis_score import load_sepsis_model, get_sepsis_score
AB_features_mean_dict={'AST': 356.2075296108291,
 'Age': 63.0167798510459,
 'Alkalinephos': 114.20330385015609,
 'BUN': 24.346708852906506,
 'BaseExcess': -0.6475370534467899,
 'Bilirubin_direct': 3.114213197969538,
 'Bilirubin_total': 2.694403177550813,
 'Calcium': 8.316976957118822,
 'Chloride': 105.7650622558037,
 'Creatinine': 1.4043820374568778,
 'DBP': 59.98580935699335,
 'FiO2': 0.5262479604121526,
 'Fibrinogen': 292.25164179104473,
 'Gender': 0.5777212530766943,
 'Glucose': 133.6092206381394,
 'HCO3': 24.09447553326941,
 'HR': 84.98526444873023,
 'Hct': 30.67489522663289,
 'Hgb': 10.582028043138731,
 'ICULOS': 27.198518124814132,
 'Lactate': 2.4692027410382145,
 'MAP': 78.76734527184637,
 'Magnesium': 2.0410037247279824,
 'O2Sat': 97.26568772153938,
 'PTT': 40.78193651125141,
 'PaCO2': 41.16614709617827,
 'Phosphate': 3.588572789252058,
 'Platelets': 199.61784112312858,
 'Potassium': 4.161507176476068,
 'Resp': 18.773459507375623,
 'SBP': 120.96235945816058,
 'SaO2': 91.21545582226761,
 'Temp': 37.02673699236573,
 'TroponinI': 9.288186528497409,
 'WBC': 11.936603760868179,
 'pH': 7.380242785410664}
def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
        data=pd.DataFrame(data,columns=column_names)

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data =data[column_names]# data[:,:-1]

    return data #df 

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
            
def fill_null_one_df(df):
    feature_name=[i for i in df.columns if i not in ["BaseExcess","Gender","EtCO2","Unit1","Unit2","HospAdmTime"]]
    for col in feature_name:
        if col=="SepsisLabel":
            continue
        """if not any(df[col]):#if all null fill mean
            df[col]=df[col].fillna(AB_features_mean_dict[col],inplace=True)"""
        else:
            """df[col].fillna(method="pad",inplace=True)#padding with before data"""
            df[col].fillna(AB_features_mean_dict[col],inplace=True) #if first is null fill mean data
    return df[feature_name]

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
   #input file
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
 
   
    # Load model.
    print('Loading sepsis model...')
    #model =joblib.load('./train56000_model.m')
    model=load_sepsis_model()
    print('Predicting sepsis labels...')
    num_files = len(files) #
    #read file
    for i, f in enumerate(files):
        print('    {}/{}...'.format(i+1, num_files))

        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)
       # print("load_after_data_type",type(data))
        #fill_data with null  return df
        data=fill_null_one_df(data)
        data=np.array(data.values)
        
        #Make predictions.
        num_rows = len(data)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        for t in range(num_rows):
            current_data = data[0:t+1]
            current_score, current_label = get_sepsis_score(current_data, model)
            scores[t] = current_score
            labels[t] = current_label

        # Save results.
        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)

    print('Done.fight_fms')
