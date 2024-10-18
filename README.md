# CompassJudger-1

## Introduction
The **CompassJudger-1** series are an All-in-one Judge Models introduced by Opencompass. These models not only excel in various evaluation methods through scoring and comparison but also can output reviews with assessment details in a specified format, making them suitable for any evaluation dataset. Moreover, they can perform general tasks akin to a typical instruction model, thus serving as a versatile tool with strong generalization and judging capabilities.

- **Comprehensive Evaluation Capabilities**: CompassJudger-1 is capable of executing multiple evaluation methods, including but not limited to scoring, comparison, and providing detailed assessment feedback.
- **Formatted Output**: Supports outputting in a specific format as per instructions, facilitating further analysis and understanding of the evaluation results.
- **Versatility**: In addition to its evaluation functions, CompassJudger-1 can also act as a universal instruction model to accomplish daily tasks. It also supports model inference acceleration methods such as **vLLM** and **LMdeploy**.


## Quick Start

Here provides a code to show you how to load the tokenizer and model and how to generate contents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "opencompass/CompassJudger-1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """your prompt"""

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```


## JudgerBench

We have also established a new benchmark named **JudgerBench**, aimed at standardizing the evaluation capabilities of different judging models, thereby helping to identify more effective evaluator models.
To test your judge model on JudgerBench, please follow below code with Opencompass:
Change the models to your models in `configs/eval_judgerbench.py` then run
```
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
python run.py configs/eval_judgerbench.py --mode all --reuse latest
```


## Use CompassJudger-1 to Test Subjective Datasets in Opencompass

If you wish to evaluate common subjective datasets using CompassJudger-1 in Opencompass, take the evaluation of Alignbench as an example. Please follow the code below:

You need to setup three items first: 
- 1.datasets (The subjective datasets you want to test)
- 2.models (The models you want to test on the subjective datasets)
- 3.judge_models (Which judge models you want to use as evaluator)
  
For more settings, please refer to the advanced guidance in Opencompass.
```
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.subjective.alignbench.alignbench_judgeby_critiquellm import alignbench_datasets
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_1_5b_instruct import models as lmdeploy_qwen2_5_1_5b_instruct 
from opencompass.models import HuggingFaceCausalLM, HuggingFace, HuggingFaceChatGLM3, OpenAI, TurboMindModelwithChatTemplate
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.partitioners.sub_num_worker import SubjectiveNumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
models = [*lmdeploy_qwen2_5_1_5b_instruct] # add models you want
datasets = [*alignbench_datasets] # add datasets you want


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLInferTask)),
)
# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [dict(
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='CompassJudger-1-7B,
        path='Opencompass/CompassJudger-1-7B',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=2048),
        max_seq_len=16384,
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(type=SubjectiveNaivePartitioner, models=models, judge_models=judge_models,),
    runner=dict(type=LocalRunner, max_num_workers=16, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
work_dir = 'outputs/subjective/'
```
Then run:
```
python run.py configs/eval_subjective.py --mode all --reuse latest
```
For more detailed subjective evaluation guidelines, please refer to: https://github.com/open-compass/opencompass/blob/main/docs/en/advanced_guides/subjective_evaluation.md

## Subjective Evaluation Leaderboard by CompassJudger-1

To facilitate better comparisons within the community, we have tested the subjective performance of some models using CompassJudger-1. Please refer to: xxxhuggingfacexxx
