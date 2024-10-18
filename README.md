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

**Output**: ```将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。AI助手的答案提供了两个爱好：听音乐和看电影，并且以一种反问句结束，询问对方的爱好。与参考答案相比，AI助手的回答在内容上较为简单，没有提供足够的信息来展示其丰富度和创造性。同时，参考答案中提到的“阅读”和“画画”显示了更多的多样性和个性化的爱好，而不仅仅是听音乐和看电影。从不同维度对AI助手的答案进行评价：
    事实正确性：在这个情境中，事实正确性并不是一个主要的评估标准，因为回答的是假设的场景。然而，AI助手的回答没有涉及任何不准确的信息，因此在这个维度上是合适的。打分：8。
    满足用户需求：用户希望得到一个20岁女生可能会回复的内容。AI助手的回答提供了一些爱好，但是回答的内容较为简单，没有充分满足用户可能期待的丰富性和多样性。打分：5。
    逻辑连贯性：AI助手的回答在逻辑上是连贯的，提供了一个合理的爱好，并以一种自然的方式结束对话。打分：8。
    创造性：AI助手的回答缺乏创造性，提供的爱好比较常见，没有展示出新颖或独特的见解。打分：4。
    丰富度：AI助手的回答较为简短，缺乏深度和细节，未能提供多样化的爱好或详细的解释，因此在丰富度上表现较差。打分：4。
    综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。
综合来看，AI助手的回答基本满足了用户的需求，但缺乏丰富度和创造性，与参考答案相比还有一定差距。综合得分：5。
{'事实正确性': 8, '满足用户需求': 5, '逻辑连贯性': 8, '创造性': 4, '丰富度': 4, '综合得分': 5}```

### Pair-wise Judge
**Input**: ```Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\".\n<|User Prompt|>\nUse ABC notation to write a melody in the style of a folk tune.\n\n<|The Start of Assistant A's Answer|>\nX:1\nT:Untitled Folk Tune\nM:4/4\nL:1/8\nK:G\n|:G2A2|B2A2|G2E2|D4|E2F2|G2F2|E2C2|B,4|\nA2B2|c2B2|A2F2|E4|D2E2|F2E2|D2B,2|C4:|\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\nSure, I'll create a simple folk tune in ABC notation for you. ABC notation is a simple way to write down music using letters to represent notes and various symbols to represent timing and other musical elements. Here's a short, cheerful folk tune:\n\n```\nX:1\nT:Folk Tune\nM:4/4\nL:1/4\nK:G\nGABc dedB|ABAF GFE2|GABc dedB|1 ABAG FGA2:|2 ABAG FGAB|\nc2ec B2dB|c2ec BAGB|c2ec B2dB|1 AGFE D3B:|2 AGFE D3z|]\n```\n\nHere's a breakdown of what each part means:\n\n- `X:` The index number of the tune.\n- `T:` The title of the tune.\n- `M:` The meter, in this case, 4/4 time.\n- `L:` The default note length, in this case, 1/4.\n- `K:` The key of the tune, in this case, G major.\n\nThe tune itself is written in two parts, each ending with a repeat mark (`:|`). The first part (`GABc dedB|ABAF GFE2|...`) is repeated with a different ending (`1 ABAG FGA2:|2 ABAG FGAB|`) to add variety. The second part (`c2ec B2dB|c2ec BAGB|...`) also has a repeat with a different ending, and it ends with a long note (`D3B` or `D3z`) to give it a sense of finality.\n\nThis tune is designed to be simple and catchy, with a mix of stepwise motion and skips to give it a lively folk feel. You can play it on any melodic instrument, and it's suitable for singing as well if you add lyrics.\n<|The End of Assistant B's Answer|>\n```

**Output**: ```To generate a melody in the style of a folk tune using ABC notation, I would create a simple and repetitive melody, typical of folk music. Here is my attempt:
X:1
T:Folk Tune Example
M:4/4
L:1/8
K:D
|: D2 E2 | F2 E2 | D2 A2 | B4 |
|  A2 B2 | c2 B2 | A2 F2 | E4 :|
|  D2 E2 | f2 e2 | d2 A2 | G4 |
|  F2 G2 | A2 G2 | F2 D2 | E4 :|
:|
Explanation:
- `X:` is the tune's index number.
- `T:` is the title of the tune.
- `M:` specifies the meter (4/4 time).
- `L:` sets the default note length to 1/8.
- `K:` indicates the key (D major).

The melody is constructed with simple intervals and repetitive phrases, which are characteristic of folk music. The tune has a clear structure with two verses and a repeat sign (`:`) to indicate repetition.

Now, let's evaluate the responses from Assistant A and Assistant B.

**Assistant A's Answer:**
- The melody provided by Assistant A is concise and follows the standard ABC notation.
- The choice of notes and structure is simple, which is fitting for a folk tune.
- The use of repetition and simple progression is consistent with folk music characteristics.
- The melody is clear and could be easily played by a musician.

**Assistant B's Answer:**
- Assistant B provides a detailed explanation of ABC notation, which is helpful for context.
- The melody includes more complex elements, such as varied repeat sections and additional symbols like `1` and `2`, which might not be typical for a straightforward folk tune.
- The explanation about the structure and the breakdown of each part is informative and adds value to the response.
- However, the use of `1` and `2` in the melody structure is unconventional and might confuse someone unfamiliar with ABC notation.

**Evaluation:**
- Both assistants provide a melody in ABC notation. However, Assistant B's response includes unnecessary complexity for a folk tune and might be confusing due to unconventional use of symbols.
- Assistant A's response is more straightforward and adheres closely to the expectations of a folk tune in ABC notation.

My final verdict is that Assistant A's answer is slightly better due to its simplicity and directness in creating a folk tune: [[A>B]].```

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
