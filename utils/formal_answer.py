import json

from gensim import logger

from utils import compute_bleu_rouge
from utils import normalize
from utils.dureader_eval import local_prf


def eval(path1 , path2):
    # with open(path2, encoding='utf-8') as f:
    #     data = json.load(f)
    # id = list(int(i) for i in data.keys())

    with open(path1 , encoding='utf-8') as f:
        pred_answers , ref_answers = [] , []
        for lidx , line in enumerate(f):
            sample = json.loads(line.strip())
            pred_answers.append({'question_id': sample['question_id'] ,
                                 'question': sample['question'] ,
                                 'question_type': sample['question_type'] ,
                                 'answers': ["No Answer Present."] ,
                                 'ref_answers': sample['ref_answers'] ,
                                 'entity_answers': [[]] ,
                                 'yesno_answers': []})
            ref_answers.append({'question_id': sample['question_id'] ,
                                'question': sample['question'] ,
                                'question_type': sample['question_type'] ,
                                'answers': sample['ref_answers'] ,
                                'ref_answers': sample['ref_answers'] ,
                                'entity_answers': [[]] ,
                                'yesno_answers': []})
    pred_dict , ref_dict = {} , {}
    F1 = 0
    count = 0
    for pred , ref in zip(pred_answers , ref_answers):
        question_id = ref['question_id']
        if len(ref['answers']) > 0:
            pred_dict[question_id] = normalize(pred['answers'])
            ref_dict[question_id] = normalize(ref['answers'])


            F = local_prf(pred['answers'][0].split() , ref['answers'][0].split())
            F1 +=F

            count += 1
    bleu_rouge = compute_bleu_rouge(pred_dict , ref_dict)
    F1_avg = F1 / count
    return bleu_rouge , F1_avg


if __name__ == '__main__':
    bleu_rouge , F1_avg = eval(path1='/home/congyao/DuReader-master/data/results/mrc_result7.8/dev.predicted.json' ,
                               path2='/home/congyao'
                                     '/Downloads/bert-master/output_mrc7.11/search_predictions.json')
    print('Dev eval result: {}'.format(bleu_rouge))
    print(bleu_rouge['Rouge-L'] , F1_avg)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('pred_file',default='/home/congyao/DuReader-master/data/results/mrc_result7.8/dev.predicted.json', help='predict file')
    # parser.add_argument('ref_file',default='/home/congyao/DuReader-master/mrc_dev.json', help='reference file')
    # parser.add_argument('task',
    #         help='task name: Main|Yes_No|All|Entity|Description')
    #
    # args = parser.parse_args()
    # args.task = args.task.lower().replace('_', '')
    # main(args)
