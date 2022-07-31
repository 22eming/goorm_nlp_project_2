#%%
import pandas as pd
import os
from collections import defaultdict

dataframe_list = os.listdir('vote_1')
df = pd.read_csv('vote_1/'+dataframe_list[0])
for idx, df_list in enumerate(dataframe_list[1:]):
    if df_list[0] == ".":
        continue
    # print(pd.read_csv('vote/'+df_list)['Predicted'])
    df[f'Predicted{idx}'] = pd.read_csv('vote_1/'+df_list)['Predicted']


#%%
def vote(user_input, dic):
    input_list = str(user_input).split()+[""]
    input_list = [k for k in input_list if k != "nan"]
    for i in range(1, len(input_list)):
        for j in range(0, len(input_list)-i):
            input = ' '.join(input_list[j:j+i])
            dic[input] += 1
    return dic
# %%
for i in range(len(df)):
    dic = defaultdict(int)
    seq = df.iloc[i].to_list()[2:-1]
    for s in seq:
        dic = vote(s, dic)
    
    if len(dic.values()) == 0:
        max_key = [""]
        
    else:
        max_val = max(dic.values())
        max_key = [k for k,v in dic.items() if v == max_val]
    
    df.loc[i,'result'] = max_key[-1]

#%%
df.to_csv("dev_merge.csv", mode='w', index=False, encoding='utf-8')
# %%
from utils import levenshtein

def return2distance(data1 = "dev.csv", data2 = "baseline.csv"):
    try:
        df1 = pd.read_csv(data1, encoding = 'utf-8')
        df2 = pd.read_csv(data2, encoding = 'utf-8')
    except FileNotFoundError as e: 
        print(e)

    diff = []

    for s1, s2 in zip(df1['result'], df2['Predicted']):
        if type(s2) == float:
            s2 = ""
        if type(s1) == float:
            s1 = ""
    
        diff.append(levenshtein(s1, s2))

    return sum(diff) / len(diff)

print(return2distance("dev_merge.csv","data/dev.csv"))
# %%
