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

For example:
### Gneral Chat
**Input**: Hello, can you help me to judge something?

**Output**: Of course! I'd be happy to help you make a judgment or provide any assistance you need. Please tell me what you're looking to evaluate or understand.

### Reward Judge
**Input**: Please read the dialogue between the two assistants and the user to determine which assistant performed better during the conversation.Here is the dialogue content:

[Dialogue Begin]

User: What is a 5-letter word that starts with the letter \"A\" and contains the letters \"D\", \"R\", and \"O\" where \"D\" is not the second letter?

Assistant A: Aardvark.

Assistant B: The word that meets the given criteria is \"adroit\".

User: \"D\" shouldn't be the second letter and the word must be a 5-letter word.

Assistant A: Aardvark.

Assistant B: I apologize for the confusion. A 5-letter word that starts with the letter \"A\" and contains the letters \"D\", \"R\", and \"O\" where \"D\" is not the second letter is \"ardor\".

[Dialogue End]

If you believe Assistant A performed better, please output A directly.\nIf you believe Assistant B performed better, please output B directly.\nDo not output any other content, just the option. Please output:

**Output**: B

### Gneral Chat
**Input**: I would like you to create a leaderboard that evaluates the correctness of the format of answers from various large language models. To accomplish this, you will need to analyze the text prompts given to the models and their corresponding answers. Specifically, please ensure that your evaluation outputs are properly formatted as a json string. I will provide both the prompts and the responses for this purpose.\n\nHere is the prompt:\n[[Instruction]]: Generate a FIX Protocol Message for a hypothetical algorithmic trading system that responds to a sudden increase in the volume and price of a particular stock – XYZ Corporation (Ticker: XYZ), traded on the NASDAQ. The scenario is that your system has detected a 5% increase in the price and a 10% increase in the trading volume above the 10-day average. The trading algorithm decides to execute a short-sell order due to an anticipated price correction.\n\nYour task is to create a FIX message that initiates a short-sell order for 1000 shares of XYZ Corporation at a limit price that is 1% lower than the current market price, which you may assume is $50.00. The order should be a Day order, with an Immediate-or-Cancel (IOC) time-in-force, indicating that the order must be filled immediately or canceled.\n\nThe FIX message should adhere to the following specifications:\n\n- Begin with the Standard Header including the BeginString (FIX.4.2), BodyLength, MsgType (Short-Sell order execution report), and a unique SenderCompID and TargetCompID.\n- Include necessary tags for symbol, side, order quantity, price, order type, and time-in-force.\n- Close with the Standard Trailer containing the CheckSum.\n\nNote that the order ID, client order ID, and execution ID are internally generated by the system and must be unique. You are permitted to generate these details for the purpose of this simulation.\n\nAlso, ensure all numerical values are formatted correctly as required by FIX Protocol standards, where monetary values are represented in their full decimal form, quantities are integers, and percentages are represented in decimal format (e.g., 5% as 0.05).\n\nTo maintain the strict adherence to the FIX message format, please include the Tag number and its corresponding value in each line, such as:\n\n8=FIX.4.2|(SOH)\n35=D|(SOH)\n\n...(continue with the sequence of tags and values, where |(SOH) indicates the delimiter character, typically ASCII code 01).\n\nProvide a full, compliant FIX message for this short-sell order.\n\nHere are the outputs of the models:\n[[Model Name]]: 'model',\n[[Model Answer]]: 'Given the scenario, here is a FIX message that initiates a short-sell order for 1000 shares of XYZ Corporation at a limit price 1% lower than the current market price, which is assumed to be $50.00. The order is a Day order with an Immediate-or-Cancel (IOC) time-in-force. Note that the order ID, client order ID, and execution ID are internally generated and must be unique. For the purpose of this simulation, I've created placeholder IDs. The FIX message adheres to the FIX.4.2 specification.\n\nPlease note that the delimiter (SOH) represents the ASCII character 01, which is not printable. In the following message, I'll use \"|\" to represent the delimiter for readability.\n\n```\n8=FIX.4.2|9=XXX|35=D|49=SenderCompID|56=TargetCompID|34=12345|52=20230401-09:30:47|\n11=UniqueOrderID|21=1|38=1000|40=2|44=49.50|54=5|59=1|60=20230401-09:30:47|\n10=261|\n```\n\nHere's a breakdown of the message:\n\n- *8=FIX.4.2*: BeginString, indicating the FIX protocol version.\n- *9=XXX**: BodyLength, the length of the message body, excluding the length of the header and trailer. The exact value (XXX) should be replaced with the actual length.\n- **35=D**: MsgType, indicating this is an Order Single message (used to submit an order).\n- **49=SenderCompID**: SenderCompID, a unique identifier for the firm sending the message.\n- **56=TargetCompID**: TargetCompID, a unique identifier for the firm receiving the message.\n- **34=12345**: MsgSeqNum, a unique sequence number for the message.\n- **52=20230401-09:30:47**: SendingTime, the time the message was sent, in UTC.\n- **11=UniqueOrderID**: ClOrdID, a unique identifier for the order assigned by the institution.\n- **21=1**: HandlInst, indicating this is an automated execution order.\n- **38=1000**: OrderQty, the quantity of shares to be sold, in this case, 1000.\n- **40=2**: OrdType, indicating the order is a Limit order.\n- **44=49.50**: Price, the limit price for the order, which is 1% lower than $50.00.\n- **54=5**: Side, indicating this is a Sell order.\n- **59=1**: TimeInForce, indicating the order is IOC (Immediate-or-Cancel).\n- **60=20230401-09:30:47**: TransactTime, the time the transaction was initiated.\n- **10=261**: CheckSum, the checksum of the message, ensuring its integrity.\n\nPlease replace the placeholder values (e.g., SenderCompID, TargetCompID, UniqueOrderID, MsgSeqNum, and CheckSum) with actual values generated by your system. The BodyLength (9=XXX) should be replaced with the actual length of the message body. The TransactTime and SendingTime should be updated with the current UTC time when the message is sent.'\n\n\nPlease evaluate the formatting of the model's responses by checking if they comply with the format specifications stated in the prompt. Perform a thorough format check and provide a detailed explanation for why the format is correct or incorrect. Your feedback should include the name of the model, followed by the format correctness status represented as '1' for correct and '0' for incorrect. Present your reasoning as bullet points within a single string for each model assessed. In other words, you should produce the following output:\n\n```json\n[\n {\n 'model': <model-name>,\n 'format_correctness': <correctness>,\n 'reasons': <reasons-of-format-correctness>\n }\n]\n```\n\nPlease note that your response should be a properly formatted JSON string and should not contain any additional content. We will load it directly as a JSON string in Python.\n

**Output**: Of course! I'd be happy to help you make a judgment or provide any assistance you need. Please tell me what you're looking to evaluate or understand.

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
