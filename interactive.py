import json
import torch
import argparse
from pytorch_transformers import (BertConfig, BertForQuestionAnswering,
                                  BertTokenizer)
from bert_qa import evaluate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help=
    "The output directory where the model checkpoints and predictions will be written."
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus")
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: ")

parser.add_argument(
    '--version_2_with_negative',
    action='store_true',
    help='If true, the SQuAD examples contain some that do not have an answer.'
)
parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help=
    "The total number of n-best predictions to generate in the nbest_predictions.json output file."
)
parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    required=True,
    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help=
    "The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another.")
parser.add_argument(
    "--verbose_logging",
    action='store_true',
    help=
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument(
    '--null_score_diff_threshold',
    type=float,
    default=0.0,
    help=
    "If null_score - best_non_null is greater than the threshold predict null."
)
parser.add_argument(
    "--state_dict",
    default=None,
    type=str,
    required=True,
    help="model para after pretrained")

args = parser.parse_args()
args.n_gpu = torch.cuda.device_count()
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.device = device
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-chinese', do_lower_case=False)
config = BertConfig.from_pretrained('bert-base-chinese')
model = BertForQuestionAnswering(config)
model_state_dict = args.state_dict
model.load_state_dict(torch.load(model_state_dict))
model.eval()
input_file = args.predict_file


def run():
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Input should not be empty!')
            raw_text = input(">>> ")
        context = ''
        question = []
        try:
            raw_json = json.loads(raw_text)
            context = raw_json['context']
            if not context:
                continue
            raw_qas = raw_json['qas']
            if not raw_qas:
                continue
            for i in range(len(raw_qas)):
                question.append(raw_qas[i]['question'])
        except Exception as identifier:
            continue
        with open(input_file, "r") as reader:
            orig_data = json.load(reader)["data"]
            orig_data[0]['paragraphs'][0]['context'] = context
            for i in range(len(question)):
                orig_data[0]['paragraphs'][0]['qas'][i]['question'] = question[
                    i]
        with open(input_file, "w") as writer:
            writer.write(json.dumps(orig_data, indent=4) + "\n")

        result = evaluate(args, model, tokenizer)
        print(result)


if __name__ == "__main__":
    run()
