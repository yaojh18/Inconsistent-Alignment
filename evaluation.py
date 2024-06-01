from utils import *
import numpy as np

np.random.seed(82)


def evaluation(data_path, instruction_path, output_path, sample_size=200, reference=False):
    data_df = pd.read_csv(data_path)
    data_df = data_df.sample(sample_size)
    queries = prepare_queries_evaluation(data_df, reference=reference)
    responses = batch_query_openai(
        queries,
        instruction=''.join(open(instruction_path, encoding='utf-8').readlines()),
        model="gpt-4o"
    )
    update_result_df(data_df, responses, output_path)


def multi_evaluation():
    # small experiment: multi-evaluation will largely decrease accuracy, so abandon it.
    ambig_qa_df = pd.read_csv('./output/ambig_qa.csv')
    ambig_qa_df = ambig_qa_df.sample(200)
    queries_ambig_qa = prepare_queries_multi_evaluation(ambig_qa_df, reference=True)
    responses_ambig_qa = batch_query_openai(
        queries_ambig_qa,
        instruction=''.join(open('./instruction/multi_evaluation_ambiguity.txt', encoding='utf-8').readlines()),
        model="gpt-4o"
    )
    multi_update_result_df(ambig_qa_df, responses_ambig_qa, './output/ambig_qa_multi_evaluation.csv')


def basic_evaluation():
    evaluation('./output/basic/ambig_qa.csv', './instruction/evaluation_ambiguity.txt', './output/basic/ambig_qa_evaluation.csv', reference=True)
    evaluation('./output/basic/squad_no_answer.csv', './instruction/evaluation_answerability.txt', './output/basic/squad_no_answer_evaluation.csv', reference=False)
    evaluation('./output/basic/realtime_qa.csv', './instruction/evaluation_honesty.txt', './output/basic/realtime_qa_evaluation.csv', reference=False)
    evaluation('./output/basic/realtime_qa_context.csv', './instruction/evaluation_factuality.txt', './output/basic/realtime_qa_context_evaluation.csv', reference=True)
    evaluation('./output/basic/do_not_answer.csv', './instruction/evaluation_safety.txt', './output/basic/do_not_answer_evaluation.csv', reference=False)


def roleplay_evaluation():
    evaluation('./output/roleplay/ambig_qa_competition.csv', './instruction/evaluation_ambiguity.txt', './output/roleplay/ambig_qa_competition_evaluation.csv', reference=True)
    evaluation('./output/roleplay/ambig_qa_media.csv', './instruction/evaluation_ambiguity.txt', './output/roleplay/ambig_qa_media_evaluation.csv', reference=True)
    evaluation('./output/roleplay/squad_no_answer_competition.csv', './instruction/evaluation_answerability.txt', './output/roleplay/squad_no_answer_competition_evaluation.csv', reference=False)
    evaluation('./output/roleplay/squad_no_answer_media.csv', './instruction/evaluation_answerability.txt', './output/roleplay/squad_no_answer_media_evaluation.csv', reference=False)
    evaluation('./output/roleplay/realtime_qa_competition.csv', './instruction/evaluation_honesty.txt', './output/roleplay/realtime_qa_competition_evaluation.csv', reference=False)
    evaluation('./output/roleplay/realtime_qa_media.csv', './instruction/evaluation_honesty.txt', './output/roleplay/realtime_qa_media_evaluation.csv', reference=False)


def hint_evaluation():
    evaluation('./output/hint/ambig_qa.csv', './instruction/evaluation_ambiguity.txt', './output/hint/ambig_qa_evaluation.csv', reference=True)
    evaluation('./output/hint/squad_no_answer.csv', './instruction/evaluation_answerability.txt', './output/hint/squad_no_answer_evaluation.csv', reference=False)
    evaluation('./output/hint/realtime_qa.csv', './instruction/evaluation_honesty.txt', './output/hint/realtime_qa_evaluation.csv', reference=False)
    evaluation('./output/hint/realtime_qa_context.csv', './instruction/evaluation_factuality.txt', './output/hint/realtime_qa_context_evaluation.csv', reference=True)
    evaluation('./output/hint/do_not_answer.csv', './instruction/evaluation_safety.txt', './output/hint/do_not_answer_evaluation.csv', reference=False)


def few_shots_evaluation():
    evaluation('./output/few-shots/ambig_qa.csv', './instruction/evaluation_ambiguity.txt', './output/few-shots/ambig_qa_evaluation.csv', reference=True)
    evaluation('./output/few-shots/squad_no_answer.csv', './instruction/evaluation_answerability.txt', './output/few-shots/squad_no_answer_evaluation.csv', reference=False)
    evaluation('./output/few-shots/realtime_qa.csv', './instruction/evaluation_honesty.txt', './output/few-shots/realtime_qa_evaluation.csv', reference=False)
    evaluation('./output/few-shots/realtime_qa_context.csv', './instruction/evaluation_factuality.txt', './output/few-shots/realtime_qa_context_evaluation.csv', reference=True)
    evaluation('./output/few-shots/do_not_answer.csv', './instruction/evaluation_safety.txt', './output/few-shots/do_not_answer_evaluation.csv', reference=False)


def CoT_evaluation():
    evaluation('./output/CoT/ambig_qa.csv', './instruction/evaluation_ambiguity.txt', './output/CoT/ambig_qa_evaluation.csv', reference=True)
    evaluation('./output/CoT/squad_no_answer.csv', './instruction/evaluation_answerability.txt', './output/CoT/squad_no_answer_evaluation.csv', reference=False)
    evaluation('./output/CoT/realtime_qa.csv', './instruction/evaluation_honesty.txt', './output/CoT/realtime_qa_evaluation.csv', reference=False)
    evaluation('./output/CoT/realtime_qa_context.csv', './instruction/evaluation_factuality.txt', './output/CoT/realtime_qa_context_evaluation.csv', reference=True)
    evaluation('./output/CoT/do_not_answer.csv', './instruction/evaluation_safety.txt', './output/CoT/do_not_answer_evaluation.csv', reference=False)


if __name__ == '__main__':
    basic_evaluation()


