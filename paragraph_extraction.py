# -*- coding:utf-8 -*-

import sys

if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('../')
import json
import copy
from preprocess import metric_max_over_ground_truths , f1_score,metric_all_ground_truths


def compute_paragraph_score(sample, mode =None):
    """
    For each paragraph, compute the f1 score compared with the question
    Args:
        sample: a sample in the dataset.
    Returns:
        None
    Raises:
        None
    """
    bleu , rouge = None, None
    if mode == "train" and "segmented_answers" in sample:
        sentence = sample["segmented_answers"]
        sentence += sample["segmented_question"]
    else:
        sentence = sample["segmented_question"]
    for doc in sample['documents']:
        doc['segmented_paragraphs_scores'] = []
        for p_idx , (para_tokens, p_tokens) in enumerate(zip(doc['segmented_paragraphs'], doc['paragraphs'])):
            if len(sentence) > 0:

                if mode == "train":
                    related_score = metric_all_ground_truths(f1_score ,bleu, rouge,
                                                                 para_tokens ,
                                                                 sentence)

                else:
                    related_score = metric_all_ground_truths(f1_score ,bleu, rouge,
                                                                  para_tokens ,
                                                                  [sentence])
            else:
                related_score = 0.0
            doc['segmented_paragraphs_scores'].append(related_score)


def compute_mrc_paragraph_score(sample, mode =None):
    """
    For each paragraph, compute the f1 score compared with the question
    Args:
        sample: a sample in the dataset.
    Returns:
        None
    Raises:
        None
    """

    question = sample["question"]
    for doc in sample['documents']:
        doc['paragraphs_scores'] = []
        for p_idx , para_tokens in enumerate(doc['paragraphs']):
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score ,
                                                              para_tokens ,
                                                              [question])
            else:
                related_score = 0.0
            doc['scores'].append(related_score)


def dup_remove(doc):
    """
    For each document, remove the duplicated paragraphs
    Args:
        doc: a doc in the sample
    Returns:
        bool
    Raises:
        None
    """
    paragraphs_his = {}
    del_ids = []
    para_id = -1
    if 'most_related_para' in doc:
        para_id = doc['most_related_para']
    doc['paragraphs_length'] = []
    for p_idx , (segmented_paragraph , paragraph_score) in \
            enumerate(zip(doc["segmented_paragraphs"] , doc["segmented_paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        if p_idx < para_id:
            prev_del_num += 1
        del doc["segmented_paragraphs"][p_idx - del_num]
        del doc["segmented_paragraphs_scores"][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1
    if len(del_ids) != 0:
        if 'most_related_para' in doc:
            doc['most_related_para'] = para_id - prev_del_num
        doc['paragraphs'] = []
        for segmented_para in doc["segmented_paragraphs"]:
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def dup_mrc_remove(doc):
    """
    For each document, remove the duplicated paragraphs
    Args:
        doc: a doc in the sample
    Returns:
        bool
    Raises:
        None
    """
    paragraphs_his = {}
    del_ids = []
    para_id = -1
    if 'most_related_para' in doc:
        para_id = doc['most_related_para']
    doc['paragraphs_length'] = []
    for p_idx , (segmented_paragraph , paragraph_score) in \
            enumerate(zip(doc["paragraphs"] , doc["paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        if p_idx < para_id:
            prev_del_num += 1
        del doc["paragraphs"][p_idx - del_num]
        del doc["paragraphs_scores"][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1
    if len(del_ids) != 0:
        if 'most_related_para' in doc:
            doc['most_related_para'] = para_id - prev_del_num
        doc['paragraphs'] = []
        for segmented_para in doc["paragraphs"]:
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def first_sentence(para):
    if not len(para):
        return []
    split_tag = ['?' ,'.', '!' ,'?', '?','?' ]
    s = []
    for word in para:
        s.append(word)
        if word in split_tag:
            break
    if s[-1] not in split_tag:
        s.append(u'<spliter>')
    return s

def paragraph_selection(sample , mode, source):
    """
    For each document, select paragraphs that includes as much information as possible
    Args:
        sample: a sample in the dataset.
        mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    Returns:
        None
    Raises:
        None
    """
    # predefined maximum length of paragraph
    MAX_P_LEN = 500
    # predefined splitter
    splitter = u''
    # topN of related paragraph to choose
    topN = 3
    topK = 3
    doc_id = None
    signal = ['how_long', 'when', 'where', 'who', 'how_much', 'why', 'what', 'which', 'how']
    sig = [['多长时间', '多久'], []]
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            return
    for d_idx , doc in enumerate(sample['documents']):
        if 'segmented_paragraphs_scores' not in doc:
            continue
        status = dup_remove(doc)
        segmented_title = doc["segmented_title"]
        title_len = len(segmented_title)
        para_id = None
        key_id = 0
        if doc_id is not None:
            para_id = sample['documents'][doc_id]['most_related_para']
        total_len = title_len + len(doc['paragraphs_length'])
        # add splitter
        para_num = len(doc["segmented_paragraphs"])
        total_len += para_num
        if total_len <= MAX_P_LEN:
            incre_len = title_len
            total_segmented_content = copy.deepcopy(segmented_title)
            for p_idx , segmented_para in enumerate(doc["segmented_paragraphs"]):

                if doc_id == d_idx and para_id > p_idx:
                    incre_len += len([splitter] + segmented_para)
                if doc_id == d_idx and para_id == p_idx:
                    incre_len += 1
                total_segmented_content += [splitter] + segmented_para
            if doc_id == d_idx:
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            #doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0
            continue
        # find topN paragraph id
        para_infos = []
        first_sentences = []
        for p_idx , (para_tokens , para_scores) in \
                enumerate(zip(doc['segmented_paragraphs'] , doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens , para_scores , len(para_tokens) , p_idx))
        topN_idx = []
        next = None
        if source == "search":
            para_infos.sort(key=lambda x: (-x[1] , x[2]))
            for i, para_info in enumerate(para_infos[:topN]):
                if para_info[-1] != next:
                    topN_idx.append(para_info[-1])
                if i == 0 and para_info[-1] != len(doc['segmented_paragraphs'])-1:
                    next = para_info[-1] + 1
                    topN_idx.append(next)
            # choose 3-10 first sentence
            for p_id , para in enumerate(para_infos):
                if topN < p_id < 10:
                    first_sentences += [splitter]+first_sentence(para[0])

        elif source == "zhidao":
            for para_info in para_infos:
                topN_idx.append(para_info[-1])

        final_idx = []
        total_len = title_len
        if doc_id == d_idx:
            if mode == "train":
                final_idx.append(para_id)
                total_len = title_len + 1 + doc['paragraphs_length'][para_id]
        for i, id in enumerate(topN_idx):
            if i == 0:
                key_id = id + 1
            if total_len > MAX_P_LEN:
                break
            if total_len < MAX_P_LEN:
                Last_Len = MAX_P_LEN - total_len
            if doc_id == d_idx and id == para_id and mode == "train":
                continue
            total_len += 1 + doc['paragraphs_length'][id]
            final_idx.append(id)
        total_segmented_content = copy.deepcopy(segmented_title)
        final_idx.sort(reverse=False)
        incre_start = title_len
        incre_len = title_len
        incre_end = total_len
        L = 0
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += 1 + doc['paragraphs_length'][id]
            if doc_id == d_idx and id == para_id:
                incre_len += 1
                incre_start = incre_len
                incre_end = incre_len + len(doc['segmented_paragraphs'][id])

            total_segmented_content += [splitter] + doc['segmented_paragraphs'][id]
        if doc_id == d_idx:
            answer_start = incre_start
            answer_end = incre_end
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end - 1
        total_segmented_content += first_sentences

        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len]
        #doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0
        
        
def mrc_paragraph_selection(sample , mode):
    
    # For each document, select paragraphs that includes as much information as possible
    # Args:
    #     sample: a sample in the dataset.
    #     mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    # Returns:
    #     None
    # Raises:
    #     None
    
    # predefined maximum length of paragraph
    MAX_P_LEN = 1000
    # predefined splitter
    splitter = u'<splitter>'
    # topN of related paragraph to choose
    topN = 3
    doc_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            return
    for d_idx , doc in enumerate(sample['documents']):
        if 'paragraphs_scores' not in doc:
            continue
        status = dup_remove(doc)
        segmented_title = doc["title"]
        title_len = len(segmented_title)
        para_id = None
        if doc_id is not None:
            para_id = sample['documents'][doc_id]['most_related_para']
        total_len = title_len + sum(doc['paragraphs_length'])
        # add splitter
        para_num = len(doc["segmented_paragraphs"])
        total_len += para_num
        if total_len <= MAX_P_LEN:
            incre_len = title_len
            total_segmented_content = copy.deepcopy(segmented_title)
            for p_idx , segmented_para in enumerate(doc["segmented_paragraphs"]):
                if doc_id == d_idx and para_id > p_idx:
                    incre_len += len([splitter] + segmented_para)
                if doc_id == d_idx and para_id == p_idx:
                    incre_len += 1
                total_segmented_content += [splitter] + segmented_para
            if doc_id == d_idx:
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0
            continue
        # find topN paragraph id
        para_infos = []
        for p_idx , (para_tokens , para_scores) in \
                enumerate(zip(doc['segmented_paragraphs'] , doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens , para_scores , len(para_tokens) , p_idx))
        para_infos.sort(key=lambda x: (-x[1] , x[2]))
        topN_idx = []
        for para_info in para_infos[:topN]:
            topN_idx.append(para_info[-1])
        final_idx = []
        total_len = title_len
        if doc_id == d_idx:
            if mode == "train":
                final_idx.append(para_id)
                total_len = title_len + 1 + doc['paragraphs_length'][para_id]
        for id in topN_idx:
            if total_len > MAX_P_LEN:
                break
            if doc_id == d_idx and id == para_id and mode == "train":
                continue
            total_len += 1 + doc['paragraphs_length'][id]
            final_idx.append(id)
        total_segmented_content = copy.deepcopy(segmented_title)
        final_idx.sort()
        incre_len = title_len
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += 1 + doc['paragraphs_length'][id]
            if doc_id == d_idx and id == para_id:
                incre_len += 1
            total_segmented_content += [splitter] + doc['segmented_paragraphs'][id]
        if doc_id == d_idx:
            answer_start = incre_len + sample['answer_spans'][0][0]
            answer_end = incre_len + sample['answer_spans'][0][1]
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end
        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len]
        doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0


if __name__ == "__main__":
    # mode="train"/"dev"/"test"
    mode = ["train","dev","test"]
    source = ['search','zhidao']
    congyao_preprocess = r'/home/congyao/preprocessed/three_1_0.5_0.5/'
    result_file =['search_train.json', 'zhidao_train.json','search_dev.json','zhidao_dev.json','search_test.json','zhidao_test.json']
    precess = r'/home/congyao/preprocessed/'
    file = ['trainset/search.train.json', 'trainset/zhidao.train.json','devset/search.dev.json', 'devset/zhidao.dev.json','testset/search.test.json', 'testset/zhidao.test.json', ]
    for i in range(6):
        print("start file",i)
        with open(congyao_preprocess+result_file[i], 'w') as fout:
            with open(precess+file[i] , 'r' , encoding='utf-8') as fin:
    
                for line in fin:
                    line = line.strip()
                    if line == "":
                        continue
                    try:
                        sample = json.loads(line , encoding='utf8')
                    except:
                        print >> sys.stderr , "Invalid input json format - '{}' will be ignored".format(line)
                        continue
                    compute_paragraph_score(sample)
                    k =0 if i%2==0 else 1
                    j =  i//2
                    paragraph_selection(sample , mode=mode[j], source=source[k])
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print("final file",i)
                
            
