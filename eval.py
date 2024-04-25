import torch
import evaluate
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel

seed = 42
set_seed(seed)

base_model_id = "meta-llama/Meta-Llama-3-8B"
dataset_name = "neil-code/dialogsum-test"


def gen(model, p, maxlen=100, sample=True):
    toks = tokenizer(p, return_tensors="pt")
    res = model.generate(
        **toks.to("cuda"),
        max_new_tokens=maxlen,
        do_sample=sample,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
    ).to("cpu")
    return tokenizer.batch_decode(res, skip_special_tokens=True)


dataset = load_dataset(dataset_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    token=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token


ft_model = PeftModel.from_pretrained(
    base_model,
    "logs/checkpoint-1000",
    torch_dtype=torch.float16,
    is_trainable=False,
)


index = 10
dialogue = dataset["test"][index]["dialogue"]
summary = dataset["test"][index]["summary"]

prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

peft_model_res = gen(
    ft_model,
    prompt,
    100,
)
peft_model_output = peft_model_res[0].split("Output:\n")[1]
prefix, success, result = peft_model_output.partition("#End")

dash_line = "-".join("" for x in range(100))
print(dash_line)
print(f"INPUT PROMPT:\n{prompt}")
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
print(dash_line)
print(f"PEFT MODEL:\n{prefix}")

original_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    token=True,
)

dialogues = dataset["test"][0:10]["dialogue"]
human_baseline_summaries = dataset["test"][0:10]["summary"]

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    human_baseline_text_output = human_baseline_summaries[idx]
    prompt = f"Instruct: Summarize the following conversation.\n{dialogue}\nOutput:\n"

    original_model_res = gen(
        original_model,
        prompt,
        100,
    )
    original_model_text_output = original_model_res[0].split("Output:\n")[1]

    peft_model_res = gen(
        ft_model,
        prompt,
        100,
    )
    peft_model_output = peft_model_res[0].split("Output:\n")[1]
    # print(peft_model_output)
    peft_model_text_output, success, result = peft_model_output.partition("#End")

    original_model_summaries.append(original_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

zipped_summaries = list(
    zip(human_baseline_summaries, original_model_summaries, peft_model_summaries)
)

df = pd.DataFrame(
    zipped_summaries,
    columns=[
        "human_baseline_summaries",
        "original_model_summaries",
        "peft_model_summaries",
    ],
)
print(df)


rouge = evaluate.load("rouge")

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0 : len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0 : len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print("ORIGINAL MODEL:")
print(original_model_results)
print("PEFT MODEL:")
print(peft_model_results)

print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

improvement = np.array(list(peft_model_results.values())) - np.array(
    list(original_model_results.values())
)
for key, value in zip(peft_model_results.keys(), improvement):
    print(f"{key}: {value*100:.2f}%")
