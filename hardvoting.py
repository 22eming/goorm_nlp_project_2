#%%
import pandas as pd
import os
from collections import defaultdict

#%%
config = {
    "data_path" : "vote_all/",
    "del_code" : ["merge","news"],
    "alpa" : 0.26,
    "dev_csv" : 'data/dev.csv',
    "best_fit" : 'merge11.csv',
    
    
}

#%%

#%%
dataframe_list = os.listdir(config['data_path'])
df = pd.read_csv(config['data_path'] + dataframe_list[0])
for idx, df_list in enumerate(dataframe_list[1:]):
    if df_list[0] == ".":
        continue
    if [1 for dc in config['del_code'] if dc in df_list]:
        continue

    df[f'Predicted{idx}'] = pd.read_csv(config['data_path']+df_list)['Predicted']
    
df['result'] = 0
#%%
# df = pd.read_csv("merge_067.csv")
#%%
def vote(user_input, dic, beta):
    input_list = str(user_input).split()+[""]
    input_list = [k for k in input_list if k != "nan"]

    for i in range(1, len(input_list)):
        for j in range(0, len(input_list)-i):
            input = ' '.join(input_list[j:j+i])
            if i == 1:
                dic[input] += 1
            elif i > 1:
                dic[input] += 1+beta
    return dic
# %%
alpa = config['alpa']
for i in range(len(df)):
    dic = defaultdict(float)
    seq = df.iloc[i].to_list()[2:-1]
    beta = 1 / (len(seq)**2 - len(seq)*alpa)
    for s in seq:
        dic = vote(s, dic, beta)
        
    if len(dic.values()) == 0:
        max_key = [""]
        
    else:
        max_val = int(max(dic.values()))
        max_key = [k for k,v in dic.items() if v >= max_val]
    
    max_sep = max([len(mk.split()) for mk in max_key])
    res = sorted([mk for mk in max_key if len(mk.split()) == max_sep], key=len)[0]

    df.loc[i,'result'] = res

#%%
import csv
with open("all_new.csv", 'w') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Id', 'Predicted'])

    rows = [[df.iloc[i]['Id'], df.iloc[i]['result']] for i in range(len(df)) ]
    
    writer.writerows(rows)
    
# df.to_csv("all_new.csv", mode='w', index=False, encoding='utf-8')
# %%
from utils import levenshtein

def return2distance(data1 = "dev.csv", data2 = "baseline.csv"):
    try:
        df1 = pd.read_csv(data1, encoding = 'utf-8')
        df2 = pd.read_csv(data2, encoding = 'utf-8')
    except FileNotFoundError as e: 
        print(e)

    diff = []
    try:
        for s1, s2 in zip(df1['Predicted'], df2['Predicted']):
            if type(s2) == float:
                s2 = ""
            if type(s1) == float:
                s1 = ""
        
            diff.append(levenshtein(s1, s2))
    except:
        for s1, s2 in zip(df1['result'], df2['Predicted']):
            if type(s2) == float:
                s2 = ""
            if type(s1) == float:
                s1 = ""
        
            diff.append(levenshtein(s1, s2))

    return sum(diff) / len(diff)

print(return2distance(config['best_fit'],"all_new.csv"))

# %%
