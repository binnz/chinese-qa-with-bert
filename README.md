<h1 align="left">基于预训练模型BERT的阅读理解</h1>


Here we are going to building a machine reading comprehension system using pretrained model bert from google, the latest advances in deep learning for NLP.

Stanford Question Answering Dataset (SQuAD) is one of the first large reading comprehension datasets in English. From the perspective of model, the inputs come in the form of a Context / Question pair, and the outputs are Answers: pairs of integers, indexing the start and the end of the answer's text contained inside the Context. 
[The 2nd Evaluation Workshop on Chinese Machine Reading Comprehension(2018)](https://github.com/ymcui/cmrc2018) release part of the datasets similar to SQuAD, which we used in this example.

The model is built on top of [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) which help to use pretrained model like BERT, GPT, GPT2 to downstream tasks. The repository includes various utilities and training scripts for multiple NLP tasks, including Question Answering. Below are two relate post about QA using bert:

>-[Understanding text with BERT](https://blog.scaleway.com/2019/understanding-text-with-bert/)

>-[Extending Google-BERT as Question and Answering model and Chatbot](https://medium.com/datadriveninvestor/extending-google-bert-as-question-and-answering-model-and-chatbot-e3e7b47b721a)

<h2 align="center">Getting Started</h2>

#### 1. Prepare data, the virtual python environment and install the package in requirements.txt

#### 2. Run the command below to fine tune

```bash
python bert_qa.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file ~/cmrc2018_train.json \
  --predict_file ~/cmrc2018_trial.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/chinese-qa-with-bert/output \
  --save_steps 200
```
#### 4. Load the fine-tuned params and run this command to interactive

```bash
python interactive.py \
  --output_dir ~/chinese-qa-with-bert/output \
  --model_type bert \
  --predict_file ~/chinese-qa-with-bert/input.json \
  --state_dict ~/chinese-qa-with-bert/output/pytorch_model.bin \
  --model_name_or_path bert-base-chinese
```

the input json string contains context and question is like this:

```json
{"context": "海梅·雷耶斯（）是DC漫画公司的一个虚拟人物。该人物首先出现于《无限危机 #3》（2006年二月），是第三代蓝甲虫，由作家基斯·吉芬和约翰·罗杰斯创作，屈伊·哈姆纳作画。海梅与他的父母妹妹生活在得克萨斯州的艾尔帕索。他的父亲拥有一间汽车修理厂。海梅建议自己帮助父亲在汽车修理厂中干活，然而他的父亲迄今为止未答应他，觉得海梅应该花更多的功夫在学习上并享受他自己的童年生活。海梅对他的家庭和朋友们有强烈的责任感，可是他经常抱怨做一个仅解决琐事的人。第二代蓝甲虫(泰德·科德)死前派遣圣甲虫去给惊奇队长送信,圣甲虫也因此留在了惊奇队长的永恒之岩里。之后惊奇队长被杀，蓝甲虫降落到了得克萨斯州的艾尔帕索，被少年海梅捡到，后来蓝甲虫在海梅睡觉时融合进他的脊椎，海梅从此成为第三代蓝甲虫，此时的蓝甲虫具备了随意变武器的能力。蓝甲虫原本是宇宙侵略组织Reach的战争工具，具有自己的思想。它曾被作为礼物赐予某星球的勇士，实际上是暗中控制他们。“卡基达(Kaji Dha)”是它的启动口令，第一代蓝甲虫加勒特就在它的影响下攻击过科德。蓝甲虫在无限危机后受到外界强大能量的影响沉睡了一年，苏醒后又想控制海梅，但被海梅的意念克制了，加上它本身存在程序故障，久而久之他与海梅成了好友。","qas": [{"question": "该漫画是由谁创作的？"}]}
```
### Below are some results for show

```json
{"AVERAGE": "27.276", "F1": "54.552", "EM": "0.000", "TOTAL": 1002, "SKIP": 0, "FILE": "~/chinese-qa-with-bert/output/predictions_.
json"}
```

```json
Please Enter:{"context": "江苏路街道是中国上海市长宁区下辖的一个街道办事处，位于长宁区最东部，东到镇宁路接邻静安区的静安寺街道，北到武定西路接邻静安区的曹家渡 街道，南到华山路接邻徐汇区的湖南路街道，西部界限为长宁路、安西路、武夷路等。面积1.52平方公里，户籍人口5.26万人，下辖13个居委会。长宁区区政府设在该街道辖区内的 愚园路（近安西路）。江苏路街道的主要街道江苏路、愚园路、长宁路、武夷路、武定西路、延安西路，均为上海公共租界越界筑路，是花园洋房集中的街区，是愚园路历史文化风 貌区的主体部分。较著名的近代建筑有中西女中（今上海市第三女子中学）、王伯群及汪精卫公馆（今长宁区少年宫）、西园公寓等。江苏路、长宁路、延安西路（高架）等经过拓 宽，已经形成上海市的交通干道。上海市轨道交通二号线经过该街道辖区，设有江苏路站。江苏路街道下辖歧山居委会、江苏居委会、万村居委会、南汪居委会、东浜居委会、愚三 居委会、曹家堰居委会、西浜居委会、福世居委会、长新居委会、华山居委会、利西居委会、北汪居委会。","qas": [{"question": "江苏路街道在上海市的什么地方？"}]}
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.61it/s]

江 苏 路 街 道 是 中 国 上 海 市 长 宁 
```


Just For Learn, More Optimization to Do