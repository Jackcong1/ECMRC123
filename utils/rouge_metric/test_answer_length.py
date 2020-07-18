import json


def test_length(path1):
    with open(path1, encoding='utf-8') as fin:
        data_length = []
        paralist = []
        question_list = []
        no_answer =0
        has_answer =0
        for lidx , line in enumerate(fin):
            sample = json.loads(line.strip())
            question = sample['question'].split(" ")
            question_list.append(len(question))
            for doc in sample['documents']:
                if doc['is_selected'] == "false":
                    continue
                else:
                    for se  in doc['paragraphs']:
                        str_list = se.split(" ")
                        paralist.append(len(str_list))

            if 'answers' in sample:
                for para in sample['answers']:
                    #print(para)
                    str_list = para.split(" ")
                    if str_list == ['No', 'Answer', 'Present.']:
                        no_answer +=1

                    else:
                        data_length.append(len(str_list))
                        has_answer +=1
            else:
                no_answer += 1

        return (question_list,data_length,paralist, no_answer,has_answer)


path1 = "/home/congyao/DuReader-master/mrc_train.json"
path2 = "/home/congyao/DuReader-master/mrc_dev.json"

question_list, data_length,paralist, no_answer,has_answer = test_length(path1)
question_list.sort(reverse=False)
percent_80 = len(question_list)  * 0.8
percent_90 = len(question_list)  * 0.99
percent_60 = len(question_list)  * 0.6
print(question_list[int(percent_60)],question_list[int(percent_80)],question_list[int(percent_90)],paralist[-1])
"""
train questionlength  7 8 15 90
dev 7 8 15 44
"""

# paralist.sort(reverse=False)
# percent_80 = len(paralist)  *0.8
# percent_90 = len(paralist)  *0.99
# percent_60 = len(paralist)  *0.6
#
# print(paralist[int(percent_60)],paralist[int(percent_80)],paralist[int(percent_90)],paralist[-1])
# """
# train 54 74 126 362
# dev 53 64 121 262
#
# """
#
#
print(no_answer)
print("##################")
print(has_answer)
print("##################")
data_length.sort(reverse=False)

percent_80 = len(data_length)  *0.8
percent_90 = len(data_length)  *0.9
percent_60 = len(data_length)  *0.6
print(data_length[int(percent_60)],data_length[int(percent_80)],data_length[int(percent_90)])


"""
train



305361
##################
517216
##################
12 21 30



dev
/home/congyao/anaconda3/envs/Tensorflow13/bin/python /home/congyao/DuReader-master/utils/rouge_metric/test_answer_length.py
45457
##################
56634
##################
15 23 31
"""