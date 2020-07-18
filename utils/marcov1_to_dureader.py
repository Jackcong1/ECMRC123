#coding=utf8

import sys
import json
import pandas as pd
from pandas.tests.groupby.test_value_counts import df


def trans(input_js):
    output_js = {}
    output_js['question'] = input_js['query']
    output_js['question_type'] = input_js['query_type']
    output_js['question_id'] = input_js['query_id']
    output_js['fact_or_opinion'] = ""
    output_js['documents'] = []
    for para_id, para in enumerate(input_js['passages']):
        doc = {}
        doc['title'] = ""
        if 'is_selected' in para:
            doc['is_selected'] = True if para['is_selected'] != 0 else False
        doc['paragraphs'] = [para['passage_text']]
        output_js['documents'].append(doc)

    if 'answers' in input_js:
        output_js['answers'] = input_js['answers']
    return output_js


if __name__ == '__main__':


    #df = pd.read_json(r'/home/congyao/MS-MARCO数据集/eval_v2.1_public.json')
    outputfile = r'/home/congyao/DuReader-master/mrc_eval.json'
    with open(outputfile, 'w') as f:
        for i, row in enumerate(df.iterrows()):
            marco_js = json.loads(row[1].to_json())
            dureader_js = trans(marco_js)
            print(i)
            f.write(json.dumps(dureader_js)+'\n')
