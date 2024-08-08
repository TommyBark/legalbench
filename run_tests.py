import datetime
import json
import math

import datasets
from litigaitor_mini.utils import load_model_and_tokenizer
from tqdm.auto import tqdm

from evaluation import evaluate
from utils import generate_prompts

config = json.load(open("test_config.json"))
TASK_NAMES = config["task_names"]
BATCH_SIZE = config["batch_size"]
model_config_path = config["model_config_path"]
prompt_prefix = config["prompt_prefix"]
padding_token = "<|endoftext|>"
iterations = config["iterations"]


def batch_answer(
    texts: list,
    max_completion_length: int = 5,
    batch_size: int = 1,
    tqdm_disable: bool = False,
) -> list:
    # Ensure the input is a list of strings
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings")

    all_outputs = []

    # Calculate the number of batches
    num_batches = math.ceil(len(texts) / batch_size)

    for i in tqdm(range(num_batches), disable=tqdm_disable):
        # Create a batch of inputs
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        # Tokenize the batch of input texts
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)

        stop_token = r"\n"
        # stop_token = "<|end|>"
        stop_token_id = tokenizer.encode(stop_token)[0]
        # Generate output for the current batch
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_completion_length + inputs["input_ids"].shape[1],
            eos_token_id=stop_token_id,
        )

        for j, output in enumerate(outputs):
            # Decode the generated output to text

            decoded_output = tokenizer.decode(output, skip_special_tokens=False)

            # remove padding tokens
            while decoded_output.startswith(padding_token):
                decoded_output = decoded_output.removeprefix(padding_token)

            # Index the output by the length of the input text, split on \n, and strip
            processed_output = (
                decoded_output[len(batch_texts[j]) :].split("\n")[0].strip()
            )

            # Append the processed output to the list of all outputs
            all_outputs.append(processed_output)

    return all_outputs


model, tokenizer, device = load_model_and_tokenizer(model_config_path)

metrics = {}
for task in TASK_NAMES:
    dataset = datasets.load_dataset("nguha/legalbench", task)

    with open(f"tasks/{task}/base_prompt.txt") as in_file:
        prompt_template = in_file.read()

    prompt_template = prompt_prefix + prompt_template
    test_df = dataset["test"].to_pandas()
    prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

    score = 0.0
    for _ in tqdm(range(iterations)):
        batch_generations = batch_answer(
            prompts, batch_size=BATCH_SIZE, tqdm_disable=True
        )
        metric = evaluate(task, batch_generations, test_df["answer"].tolist())
        score += metric
    metrics[task] = score / iterations

current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M")
filename = f"results_{current_datetime}.txt"

with open(filename, "w") as out_file:
    out_file.write(f"Average of {iterations} runs\n")
    for task, metric in metrics.items():
        out_file.write(f"{task}: {metric}\n")
