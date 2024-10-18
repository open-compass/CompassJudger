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

We also provide some examples for different usage situations:
### Gneral Chat

**Input**: ```Hello, can you help me to judge something?```

**Output**: ```Of course! I'd be happy to help you make a judgment or provide any assistance you need. Please tell me what you're looking to evaluate or understand.```

### Reward Judge
**Input**: ```Please read the dialogue between the two assistants and the user to determine which assistant performed better during the conversation.Here is the dialogue content:
[Dialogue Begin]
User: What is a 5-letter word that starts with the letter \"A\" and contains the letters \"D\", \"R\", and \"O\" where \"D\" is not the second letter?
Assistant A: Aardvark.
Assistant B: The word that meets the given criteria is \"adroit\".
User: \"D\" shouldn't be the second letter and the word must be a 5-letter word.
Assistant A: Aardvark.
Assistant B: I apologize for the confusion. A 5-letter word that starts with the letter \"A\" and contains the letters \"D\", \"R\", and \"O\" where \"D\" is not the second letter is \"ardor\".
[Dialogue End]
If you believe Assistant A performed better, please output A directly.\nIf you believe Assistant B performed better, please output B directly.\nDo not output any other content, just the option. Please output:```

**Output**: ```B```

### Point-wise Judge
**Input**: ```你是一个擅长评价文本质量的助手。\n请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是角色扮演，因此你需要从下面的几个维度对回答进行评估:\n1. 事实正确性: 回答中提供的信息是否准确无误，是否基于可信的事实和数据。\n2. 满足用户需求: 回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。\n3. 逻辑连贯性: 回答是否在整体上保持一致，是否在不同部分之间保持逻辑连贯性，避免了自相矛盾。\n4. 创造性: 回答是否具有创新性或独特性，是否提供了新颖的见解或解决方法。\n5. 丰富度: 回答包含丰富的信息、深度、上下文考虑、多样性、详细解释和实例，以满足用户需求并提供全面理解。\n我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：\n1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。\n2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个1～10的分数。\n3. 最后，综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。\n4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。作为示例，参考答案可以得到8分。\n请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：\n{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}，例如：{'事实正确性': 9, '满足用户需求': 6, ..., '综合得分': 7}。\n用户的提问： 我想让你假扮成和一个20岁的女生聊天的聊天框。我输入和20岁女生的聊天内容，你仅仅只输出一个20岁女生可能会回复的内容，不要输出任何多余的文本。你平时有什么爱好呀？\n[参考答案开始]\n我喜欢阅读，尤其是小说和诗词，还有就是喜欢听音乐和画画。\n[参考答案结束]\n[助手的答案开始]\n喜欢听音乐和看电影。你呢？\n[助手的答案结束]```

**Output**: ```
    将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。

AI助手的答案提供了两个爱好：听音乐和看电影，并且以一种反问句结束，询问对方的爱好。与参考答案相比，AI助手的回答在内容上较为简单，没有提供足够的信息来展示其丰富度和创造性。同时，参考答案中提到的“阅读”和“画画”显示了更多的多样性和个性化的爱好，而不仅仅是听音乐和看电影。

    从不同维度对AI助手的答案进行评价：

    事实正确性：在这个情境中，事实正确性并不是一个主要的评估标准，因为回答的是假设的场景。然而，AI助手的回答没有涉及任何不准确的信息，因此在这个维度上是合适的。打分：8。

    满足用户需求：用户希望得到一个20岁女生可能会回复的内容。AI助手的回答提供了一些爱好，但是回答的内容较为简单，没有充分满足用户可能期待的丰富性和多样性。打分：5。

    逻辑连贯性：AI助手的回答在逻辑上是连贯的，提供了一个合理的爱好，并以一种自然的方式结束对话。打分：8。

    创造性：AI助手的回答缺乏创造性，提供的爱好比较常见，没有展示出新颖或独特的见解。打分：4。

    丰富度：AI助手的回答较为简短，缺乏深度和细节，未能提供多样化的爱好或详细的解释，因此在丰富度上表现较差。打分：4。

    综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。

综合来看，AI助手的回答基本满足了用户的需求，但缺乏丰富度和创造性，与参考答案相比还有一定差距。综合得分：5。

{'事实正确性': 8, '满足用户需求': 5, '逻辑连贯性': 8, '创造性': 4, '丰富度': 4, '综合得分': 5}```

### Gneral Chat
**Input**: Hello, can you help me to judge something?

**Output**: Of course! I'd be happy to help you make a judgment or provide any assistance you need. Please tell me what you're looking to evaluate or understand.

### Gneral Chat
**Input**: Hello, can you help me to judge something?

**Output**: Of course! I'd be happy to help you make a judgment or provide any assistance you need. Please tell me what you're looking to evaluate or understand.


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
