from litigaitor_mini.utils import load_model_and_tokenizer
import math
from evaluation import evaluate
import numpy as np
from tqdm.auto import tqdm
import datasets

from tasks import TASKS, ISSUE_TASKS
from utils import generate_prompts

TASK_NAMES = ["abercrombie"]

def batch_answer(texts: list, max_completion_length: int = 5, batch_size: int = 1) -> list:
    # Ensure the input is a list of strings
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings")

    all_outputs = []
    
    # Calculate the number of batches
    num_batches = math.ceil(len(texts) / batch_size)

    for i in tqdm(range(num_batches)):
        # Create a batch of inputs
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        
        # Tokenize the batch of input texts
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt")

        # Generate output for the current batch
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_completion_length + inputs['input_ids'].shape[1],
            eos_token_id=stop_token_id
        )
        #         # Decode the generated outputs to text
        # decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # # Append the decoded outputs to the list of all outputs
        # all_outputs.extend(decoded_outputs)


        for j, output in enumerate(outputs):
            # Decode the generated output to text
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)

            # Index the output by the length of the input text, split on \n, and strip
            processed_output = decoded_output[len(batch_texts[j]):].split("\n")[0].strip()

            # Append the processed output to the list of all outputs
            all_outputs.append(processed_output)

    return all_outputs


metrics = {}
for task in TASK_NAMES:
    dataset = datasets.load_dataset("nguha/legalbench", task)

    with open(f"tasks/{task}/base_prompt.txt") as in_file:
        prompt_template = in_file.read()

    test_df = dataset["test"].to_pandas()
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

    model_config_path = "/root/LitigAItor-mini/configs/model_config.yml"
    model, tokenizer, device = load_model_and_tokenizer(model_config_path)
    stop_token = r"\n"
    stop_token_id = tokenizer.encode(stop_token)[0]


    batch_generations = batch_answer(prompts,batch_size = 3)

    metric = evaluate("abercrombie", batch_generations, test_df["answer"].tolist())

    metrics[task] = metric

with open("results.txt", "w") as out_file:
    for task, metric in metrics.items():
        out_file.write(f"{task}: {metric}\n")