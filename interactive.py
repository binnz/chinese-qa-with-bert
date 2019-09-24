import json
import torch

from pytorch_transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)
from bert-qa import evaluate



parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
args.n_gpu = torch.cuda.device_count()
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
arg.device = device
parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
config = BertConfig.from_pretrained('bert-base-chinese')
model = BertForQuestionAnswering(config)
model_state_dict = "pytorch_model.bin"
model.load_state_dict(torch.load(model_state_dict))
model.eval()
input_file = "/Users/danbin/ml-project/chinese-qa-with-bert/input.json"

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
                question.append(qas[i]['question'])
        except Exception as identifier:
            continue
        with open(input_file, "r") as reader:
            orig_data = json.load(reader)["data"]
            orig_data[0]['paragraphs'][0]['context']=context
            for i in range(len(question)):
                orig_data[0]['paragraphs'][0]['qas'][i]['question']=question[i]
        with open(input_file, "w") as writer:
            writer.write(json.dumps(orig_data, indent=4) + "\n")
        


        result = qa.evalute(arg, model, tokenizer)
        print(result)



if __name__ == "__main__":
    run(output_dir="")
