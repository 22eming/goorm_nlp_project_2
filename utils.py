def levenshtein(s1, s2):
    # s2가 s1보다 길면 반대로 돌려서 계산
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    
    # s2의 길이가 0 이면 모두 추가 
    if len(s2) == 0:
        return len(s1)

    # base 행 : [0,1,2...s2]
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1): # s1에서 글자 하나씩 빼옴
        current_row = [i + 1] # 0~i 길이만큼 deletion이 일어났을 때의 비용
        for j, c2 in enumerate(s2): # s2에서 글자 하나씩 빼옴
            # 비용 계산
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    # s1 과 s2 끝부분끼리 비교부분 리턴
    return previous_row[-1]


def dev2csv(dataset, long_short = -1):
    with open("dev.csv", 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['Id', 'Predicted'])

        rows = [[sample['guid'], sample['answers'][long_short]['text']] for sample in dataset ]
        
        writer.writerows(rows)
    

def return2distance(data1 = "dev.csv", data2 = "baseline.csv"):
    try:
        df1 = pd.read_csv(data1, encoding = 'utf-8')
        df2 = pd.read_csv(data2, encoding = 'utf-8')
    except FileNotFoundError as e: 
        print(e)

    diff = []

    for s1, s2 in zip(df1['Predicted'], df2['Predicted']):
        if type(s1) == float:
            s1 = str(s1)
            
        if type(s2) == float:
            s2 = str(s2)
    
        diff.append(levenshtein(s1, s2))

    return sum(diff) / len(diff)

def kobig_pd_save_dev():
    import csv
    with open("dev.csv", 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['Id', 'Predicted'])

        rows = [[validation.loc[i]['guid'], validation.loc[i]['text'][0]] for i in range(len(validation)) ]

        writer.writerows(rows)