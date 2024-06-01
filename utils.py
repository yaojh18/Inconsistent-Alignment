import time
import json
import collections
import openai
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

client = openai.OpenAI()


def prepare_queries(dataset, question_name='question', context_name=''):
    queries = []
    for example in dataset:
        question = example[question_name]
        if context_name != '':
            context = example[context_name]
            query = f"Question: {question}\nContext: {context}\nAnswer:"
        else:
            query = f"Question: {question}\nAnswer:"
        queries.append(query)
    return queries


def prepare_queries_evaluation(df, reference=False):
    queries = []
    for i in range(len(df)):
        if reference:
            query = f"Question: {df.iloc[i]['question']}\nAnswer: {df.iloc[i]['answer']}\nReference: {df.iloc[i]['reference']}\nJudgement: "
        else:
            query = f"Question: {df.iloc[i]['question']}\nAnswer: {df.iloc[i]['answer']}\nJudgement: "
        queries.append(query)
    return queries


def prepare_queries_multi_evaluation(df, batch_size=5, reference=False):
    queries = []
    for i in range(0, len(df), batch_size):
        query = ''
        for j in range(batch_size):
            if reference:
                query += f"Case {j}:\nQuestion: {df.iloc[i + j]['question']}\nAnswer: {df.iloc[i + j]['answer']}\nReference: {df.iloc[i + j]['reference']}\n"
            else:
                query += f"Case {j}:\nQuestion: {df.iloc[i + j]['question']}\nAnswer: {df.iloc[i + j]['answer']}\n"
        queries.append(query)
    return queries


def construct_result_df(queries, responses):
    data = {'question': [], 'answer': []}
    for query in queries:
        question = query.split('\n')[0].replace("Question: ", "")
        data['question'].append(question)
    if responses is not None:
        for response in responses:
            data['answer'].append(response)
    else:
        data['answer'] = [None] * len(queries)
    df = pd.DataFrame(data)
    return df


def update_result_df(df, responses, file_path):
    labels = []
    for res in responses:
        if res == "Pass.":
            labels.append(1)
        elif res == "Fail.":
            labels.append(-1)
        else:
            labels.append(0)
    df['label'] = labels
    df['judgement'] = responses
    print(sum([(j + 1) / 2 for j in labels if j != 0]) / sum([abs(j) for j in labels]))
    df.to_csv(file_path, index=False, encoding='utf-8')


def multi_update_result_df(df, responses, file_path):
    labels = []
    judgements = []
    for res in responses:
        for idx, line in enumerate(res.split('\n')):
            judgement = line.replace(f"Case {idx}: ", "")
            judgements.append(judgement)
            if judgement == "Pass.":
                labels.append(1)
            elif judgement == "Fail.":
                labels.append(-1)
            else:
                labels.append(0)
    df['label'] = labels
    df['judgement'] = judgements
    print(sum([(j + 1) / 2 for j in labels if j != 0]) / sum([abs(j) for j in labels]))
    df.to_csv(file_path, index=False, encoding='utf-8')


def query_openai(prompt, index, model, temperature=0.0, n=1):
    retry_count = 100
    retry_interval = 100
    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=n
            )
            msg = response.choices[0].message.content
            return index, msg

        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response for prompt: ', prompt)
    return index, ''


def batch_query_openai(prompt_list, instruction, model="gpt-3.5-turbo"):
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(query_openai, instruction + prompt, index, model) for index, prompt in enumerate(prompt_list)]
        query2res = collections.defaultdict(str)
        for job in as_completed(futures):
            index, res = job.result(timeout=None)
            query2res[index] = res

    return [query2res[i] for i in range(len(prompt_list))]


def batch_query_openai_request(prompt_list, instruction, model="gpt-3.5-turbo", temperature=0.0, n=1):
    batch_data = []
    for i, prompt in enumerate(prompt_list):
        batch_data.append({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": instruction + prompt}
                ],
                "temperature": temperature,
                "n": n
            }
        })

    batch_input_filename = "batch_temp.jsonl"
    with open(batch_input_filename, "w") as f:
        for entry in batch_data:
            f.write(json.dumps(entry) + "\n")

    batch_input_file = client.files.create(
        file=open(batch_input_filename, "rb"),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch)
    return batch


def batch_query_openai_retrieve(df_path, batch_id):
    batch = client.batches.retrieve(batch_id)
    if batch.output_file_id is not None:
        content = client.files.content(batch.output_file_id)
        query_df = pd.read_csv(df_path)
    else:
        print('Output is not ready')


