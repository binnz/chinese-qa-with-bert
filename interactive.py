import json
import torch
import argparse
from pytorch_transformers import (BertConfig, BertForQuestionAnswering,
                                  BertTokenizer)
from bert_qa import evaluate
import os

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument(
    "--train_file",
    default=None,
    type=str,
    required=False,
    help="SQuAD json for training. E.g., train-v1.1.json")
parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    required=True,
    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: ")
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: ")
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help=
    "The output directory where the model checkpoints and predictions will be written."
)

## Other parameters
parser.add_argument(
    "--config_name",
    default="",
    type=str,
    help="Pretrained config name or path if not the same as model_name")
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3"
)

parser.add_argument(
    '--version_2_with_negative',
    action='store_true',
    help='If true, the SQuAD examples contain some that do not have an answer.'
)
parser.add_argument(
    '--null_score_diff_threshold',
    type=float,
    default=0.0,
    help=
    "If null_score - best_non_null is greater than the threshold predict null."
)

parser.add_argument(
    "--max_seq_length",
    default=384,
    type=int,
    help=
    "The maximum total input sequence length after WordPiece tokenization. Sequences "
    "longer than this will be truncated, and sequences shorter than this will be padded."
)
parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help=
    "When splitting up a long document into chunks, how much stride to take between chunks."
)
parser.add_argument(
    "--max_query_length",
    default=64,
    type=int,
    help=
    "The maximum number of tokens for the question. Questions longer than this will "
    "be truncated to this length.")
parser.add_argument(
    "--do_train", action='store_true', help="Whether to run training.")
parser.add_argument(
    "--do_eval",
    action='store_true',
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--evaluate_during_training",
    action='store_true',
    help="Rul evaluation during training at each logging step.")
parser.add_argument(
    "--do_lower_case",
    action='store_true',
    help="Set this flag if you are using an uncased model.")

parser.add_argument(
    "--per_gpu_train_batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.")
parser.add_argument(
    '--gradient_accumulation_steps',
    type=int,
    default=1,
    help=
    "Number of updates steps to accumulate before performing a backward/update pass."
)
parser.add_argument(
    "--weight_decay",
    default=0.0,
    type=float,
    help="Weight deay if we apply some.")
parser.add_argument(
    "--adam_epsilon",
    default=1e-8,
    type=float,
    help="Epsilon for Adam optimizer.")
parser.add_argument(
    "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs",
    default=3.0,
    type=float,
    help="Total number of training epochs to perform.")
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help=
    "If > 0: set total number of training steps to perform. Override num_train_epochs."
)
parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help=
    "The total number of n-best predictions to generate in the nbest_predictions.json output file."
)
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
    '--logging_steps', type=int, default=50, help="Log every X updates steps.")
parser.add_argument(
    '--save_steps',
    type=int,
    default=50,
    help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action='store_true',
    help=
    "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
)
parser.add_argument(
    "--no_cuda",
    action='store_true',
    help="Whether not to use CUDA when available")
parser.add_argument(
    '--overwrite_output_dir',
    action='store_true',
    help="Overwrite the content of the output directory")
parser.add_argument(
    '--overwrite_cache',
    action='store_true',
    help="Overwrite the cached training and evaluation sets")
parser.add_argument(
    '--seed', type=int, default=42, help="random seed for initialization")

parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus")
parser.add_argument(
    '--fp16',
    action='store_true',
    help=
    "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
)
parser.add_argument(
    '--fp16_opt_level',
    type=str,
    default='O1',
    help=
    "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html")
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
model.to(args.device)
model.eval()
input_file = args.predict_file


def handle_file(input_file, context, question):
    with open(input_file, "r") as reader:
        orig_data = json.load(reader)
        orig_data["data"][0]['paragraphs'][0]['context'] = context
        for i in range(len(question)):
            orig_data["data"][0]['paragraphs'][0]['qas'][i][
                'question'] = question[i]
    with open(input_file, "w") as writer:
        writer.write(json.dumps(orig_data, indent=4) + "\n")


def run():
    while True:
        raw_text = input("Please Enter:")
        while not raw_text:
            print('Input should not be empty!')
            raw_text = input("Please Enter:")
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
            print(identifier)
            continue
        handle_file(input_file, context, question)
        evaluate(args, model, tokenizer)

        predict_file = os.path.join(args.output_dir, "predictions_.json")
        with open(predict_file, "r") as reader:
            orig_data = json.load(reader)
            print(orig_data[""])
        # clean input file
        handle_file(input_file, "", ["", "", ""])


if __name__ == "__main__":
    run()
