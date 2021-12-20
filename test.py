def changeText(char):
    if (char == 'A') or (char == 'a') or (char == '0'):
        return 0
    elif (char == 'B') or (char == 'b') or (char == '1'):
        return 1
    elif (char == 'C') or (char == 'c') or (char == '2'):
        return 2
    elif (char == 'D') or (char == 'd') or (char == '3'):
        return 3
    elif (char == 'E') or (char == 'e') or (char == '4'):
        return 4
    else:
        return 5

ad = "0:3,5:4,3:0, 3:4, 5:6, 7:9, 3:6"
ad = ad.split(',')
result = dict()
for i, v in enumerate(ad):
    result[i] = changeText(v.split(':')[1].strip())


print(result)

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
print(ANSWER_KEY)

