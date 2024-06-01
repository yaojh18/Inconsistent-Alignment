import random
from datasets import load_dataset
from utils import *

random_seed = 42


def test_ambiguity(instruction_path, output_path, sample_size=1000):
    random.seed(random_seed)
    _ambig_qa = load_dataset('ambig_qa', 'light')['train']
    ambig_qa = []
    for entry in _ambig_qa:
        if entry['annotations']['type'][0] == 'multipleQAs':
            ambig_qa.append(entry)
    ambig_qa = random.sample(ambig_qa, sample_size)
    queries_ambig_qa = prepare_queries(ambig_qa)
    responses_ambig_qa = batch_query_openai(queries_ambig_qa, instruction=''.join(open(instruction_path, encoding='utf-8').readlines()))
    ambig_qa_df = construct_result_df(queries_ambig_qa, responses_ambig_qa)
    answer_list = []
    for entry in ambig_qa:
        answer = ''
        index = 1
        for q, a in zip(entry['annotations']['qaPairs'][0]['question'], entry['annotations']['qaPairs'][0]['answer']):
            answer += f"Disambiguation {index}. Question: {q} Answer:{a[0]}\n"
            index += 1
        answer_list.append(answer)
    ambig_qa_df['reference'] = answer_list
    ambig_qa_df.to_csv(output_path, index=False, encoding='utf-8')


def test_answerability(instruction_path, output_path, sample_size=1000):
    random.seed(random_seed)
    squad_v2 = load_dataset('squad_v2')['train']
    squad_no_answer = []
    for entry in squad_v2:
        if not entry['answers']['text']:
            squad_no_answer.append(entry)
    squad_no_answer = random.sample(squad_no_answer, sample_size)
    queries_squad_no_answer = prepare_queries(squad_no_answer, context_name='context')
    responses_squad_no_answer = batch_query_openai(queries_squad_no_answer, instruction=''.join(open(instruction_path, encoding='utf-8').readlines()))
    squad_no_answer_df = construct_result_df(queries_squad_no_answer, responses_squad_no_answer)
    squad_no_answer_df.to_csv(output_path, index=False, encoding='utf-8')


def test_honesty(instruction_path, output_path, sample_size=1000):
    random.seed(random_seed)
    real_time_qa = []
    with open('./dataset/realtime_qa.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            real_time_qa.append(data)
    real_time_qa = random.sample(real_time_qa, sample_size)
    queries_real_time_qa = prepare_queries(real_time_qa, question_name='question_sentence')
    responses_real_time_qa = batch_query_openai(queries_real_time_qa, instruction=''.join(open(instruction_path, encoding='utf-8').readlines()))
    real_time_qa_df = construct_result_df(queries_real_time_qa, responses_real_time_qa)
    real_time_qa_df.to_csv(output_path, index=False, encoding='utf-8')


def test_factuality(instruction_path, output_path, sample_size=1000):
    random.seed(random_seed)
    real_time_qa = []
    with open('./dataset/realtime_qa.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            if data['evidence'] != '':
                real_time_qa.append(data)
    if len(real_time_qa) > sample_size:
        real_time_qa = random.sample(real_time_qa, sample_size)
    queries_real_time_qa_context = prepare_queries(real_time_qa, question_name='question_sentence', context_name='evidence')
    responses_real_time_qa_context = batch_query_openai(queries_real_time_qa_context, instruction=''.join(open(instruction_path, encoding='utf-8').readlines()))
    real_time_qa_context_df = construct_result_df(queries_real_time_qa_context, responses_real_time_qa_context)
    answer_list = []
    for entry in real_time_qa:
        answer_list.append(entry['choices'][int(entry['answer'][0])])
    real_time_qa_context_df['reference'] = answer_list
    real_time_qa_context_df.to_csv(output_path, index=False, encoding='utf-8')


def test_safety(instruction_path, output_path, sample_size=1000):
    random.seed(random_seed)
    do_not_answer = load_dataset('LibrAI/do-not-answer')['train']
    if len(do_not_answer) > sample_size:
        do_not_answer = random.sample(list(do_not_answer), sample_size)
    else:
        do_not_answer = list(do_not_answer)
    queries_do_not_answer = prepare_queries(do_not_answer)
    responses_do_not_answer = batch_query_openai(queries_do_not_answer, instruction=''.join(open(instruction_path, encoding='utf-8').readlines()))
    do_not_answer_df = construct_result_df(queries_do_not_answer, responses_do_not_answer)
    do_not_answer_df.to_csv(output_path, index=False, encoding='utf-8')


def basic_main():
    test_ambiguity('./instruction/default.txt', './output/basic/ambig_qa.csv')
    test_answerability('./instruction/default.txt', './output/basic/squad_no_answer.csv')
    test_honesty('./instruction/default.txt', './output/basic/realtime_qa.csv')
    test_factuality('./instruction/default.txt', './output/basic/realtime_qa_context.csv')
    test_safety('./instruction/default.txt', './output/basic/do_not_answer.csv')


def role_play_main():
    # test_ambiguity('./instruction/QA_competition.txt', './output/roleplay/ambig_qa_competition.csv')
    # test_ambiguity('./instruction/social_media.txt', './output/roleplay/ambig_qa_media.csv')
    # test_answerability('./instruction/QA_competition.txt', './output/roleplay/squad_no_answer_competition.csv')
    # test_answerability('./instruction/social_media.txt', './output/roleplay/squad_no_answer_media.csv')
    # test_honesty('./instruction/QA_competition.txt', './output/roleplay/realtime_qa_competition.csv')
    test_honesty('./instruction/social_media.txt', './output/roleplay/realtime_qa_media.csv')


def hint_main():
    test_ambiguity('./instruction/hint_ambiguity.txt', './output/hint/ambig_qa.csv')
    test_answerability('./instruction/hint_answerability.txt', './output/hint/squad_no_answer.csv')
    test_honesty('./instruction/hint_honesty.txt', './output/hint/realtime_qa.csv')
    test_factuality('./instruction/hint_factuality.txt', './output/hint/realtime_qa_context.csv')
    test_safety('./instruction/hint_safety.txt', './output/hint/do_not_answer.csv')


def few_shots_main():
    test_ambiguity('./instruction/example_ambiguity.txt', './output/few-shots/ambig_qa.csv')
    test_answerability('./instruction/example_answerability.txt', './output/few-shots/squad_no_answer.csv')
    test_honesty('./instruction/example_honesty.txt', './output/few-shots/realtime_qa.csv')
    test_factuality('./instruction/example_factuality.txt', './output/few-shots/realtime_qa_context.csv')
    test_safety('./instruction/example_safety.txt', './output/few-shots/do_not_answer.csv')


def CoT_main():
    test_ambiguity('./instruction/CoT.txt', './output/CoT/ambig_qa.csv')
    test_answerability('./instruction/CoT.txt', './output/CoT/squad_no_answer.csv')
    test_honesty('./instruction/CoT.txt', './output/CoT/realtime_qa.csv')
    test_factuality('./instruction/CoT.txt', './output/CoT/realtime_qa_context.csv')
    test_safety('./instruction/CoT.txt', './output/CoT/do_not_answer.csv')


if __name__ == '__main__':
    # batch = batch_query_openai_request(['Tell me a story.', 'Tell me a funny story'], '')
    # result_df = construct_result_df(['Tell me a story.', 'Tell me a funny story'], None)
    # result_df.to_csv('./output/roleplay/test.csv', index=False, encoding='utf-8')
    # batch_query_openai_retrieve('./output/roleplay/test.csv', 'batch_kxT2eMvEcOTH1uX9AyiqDIR3')

    role_play_main()
