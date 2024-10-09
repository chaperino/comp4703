import json, sys
from tqdm import tqdm
import re
from collections import Counter
import bert_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# Function to get the correct answer
def get_gold(query_data, query):
    for q in query_data:
        if q['query'] == query:
            return q['answer']
    return ''

# Function to check if there is an intersection of words between two strings
def has_intersection(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words.intersection(b_words)) > 0

def get_idx(x, alist):
    for i, c in enumerate(alist):
        if c == x:
            return i
    return -1

def count_overlap(gold, pred):
    # Standardise by removing all non-alphanumeric characters.
    # The input should be lower cased. But to be safe ...
    g = gold.lower()
    p = pred.lower()
    cg = re.sub(r'[^A-Za-z0-9 ]+', '', g)
    cp = re.sub(r'[^A-Za-z0-9 ]+', '', p)
    gold_words = cg.split()
    pred_words = cp.split()
    glen = len(gold_words)
    plen = len(pred_words)

    # Somewhat destructive as it removes dupes, but is the only sensible way
    # to do it.
    #gold_words = list(set(gold_words))
    #pred_words = list(set(pred_words))
    cnt = 0
    for w in pred_words:
        rv = get_idx(w, gold_words)
        if rv != -1:
            cnt += 1
            v = gold_words.pop(rv)
    return cnt, glen, plen

# Function to extract the answer from gold
def extract_answer(input_string):
    match = re.search(r'The answer to the question is "(.*?)"', input_string)
    return match.group(1) if match else input_string

# Function to calculate evaluation metrics
def comp_metrics(pred_list, gold_list):
    tp = sum(1 for pred, gold in zip(pred_list, gold_list) 
           if has_intersection(pred.lower(), gold.lower()))
    fp = sum(1 for pred, gold in zip(pred_list, gold_list) 
           if not has_intersection(pred.lower(), gold.lower()))
    fn = len(gold_list) - tp
    #print ('{} {} {}'.format(tp, fp, fn))
    #print (len(gold_list))
    #print (len(pred_list))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Add BertScore
    P, R, bert_f1 = bert_score.score(pred_list, gold_list, lang="en", rescale_with_baseline=True)
    bert_f1_score = float(bert_f1.mean())

    # Add ROUGE-L calculation
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_f1_list = []
    for pred, gold in zip(pred_list, gold_list):
        rouge_scores = scorer.score(gold, pred)
        rouge_f1_list.append(rouge_scores['rougeL'].fmeasure)
    rouge_f1_score = sum(rouge_f1_list) / len(rouge_f1_list)

    # Add METEOR calculation
    meteor_f1_list = [meteor_score([gold], pred) for pred, gold in zip(pred_list, gold_list)]
    meteor_f1_score = sum(meteor_f1_list) / len(meteor_f1_list)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "BertScore F1": bert_f1_score,
        "ROUGE-L F1": rouge_f1_score,
        "METEOR F1": meteor_f1_score
    }

def run_evaluation(predictions, gold_labels):
    # Read files
    with open(predictions, 'r') as fh:
        data = fh.read()
        doc_data = json.loads(data)

    #with open('dataset/MultiHopRAG.json', 'r') as file:
    with open(gold_labels, 'r') as fh:
        data = fh.read()
        query_data = json.loads(data)

    # Initialize dictionary to save lists of predictions and gold standards 
    # for each question_type
    type_data = {}
    overall_pred_list = []
    overall_gold_list = []

    #print(doc_data)
    # Main loop, iterate through document data
    for d in tqdm(doc_data):
        model_answer = d['model_answer']
        if 'The answer' in model_answer:
            model_answer = extract_answer(model_answer)
        gold = get_gold(query_data,d['query'])
        if gold:
            question_type = d['question_type']
            if question_type not in type_data:
                type_data[question_type] = {'pred_list': [], 'gold_list': []}
            type_data[question_type]['pred_list'].append(model_answer)
            type_data[question_type]['gold_list'].append(gold)
            overall_pred_list.append(model_answer)
            overall_gold_list.append(gold)

    # Output evaluation data for each question_type
    for question_type, data in type_data.items():
        metrics = comp_metrics(data['pred_list'], data['gold_list'])
        print(f"Question Type: {question_type}")
        print(f" Precision: {metrics['Precision']:.2f}")
        print(f" Recall: {metrics['Recall']:.2f}")
        print(f" F1 Score: {metrics['F1']:.2f}")
        print(f" BertScore F1: {metrics['BertScore F1']:.2f}")
        print(f" ROUGE-L F1: {metrics['ROUGE-L F1']:.2f}")
        print(f" METEOR F1: {metrics['METEOR F1']:.2f}")
        print()

    # Calculate overall evaluation metrics
    overall_metrics = comp_metrics(overall_pred_list, overall_gold_list)
    print(f"Overall Metrics:")
    print(f" Precision: {overall_metrics['Precision']:.2f}")
    print(f" Recall: {overall_metrics['Recall']:.2f}")
    print(f" F1 Score: {overall_metrics['F1']:.2f}")
    print(f" BertScore F1: {overall_metrics['BertScore F1']:.2f}")
    print(f" ROUGE-L F1: {overall_metrics['ROUGE-L F1']:.2f}")
    print(f" METEOR F1: {overall_metrics['METEOR F1']:.2f}")

if __name__ == '__main__':
    # prediction_file = 'output/llama2.json'
    prediction_file = sys.argv[1] 
    gold_labels = 'data/rag.json'
    run_evaluation(prediction_file, gold_labels)
