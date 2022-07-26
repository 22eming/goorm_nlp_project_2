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
