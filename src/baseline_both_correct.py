# Test clean 5-shot country-capital recall on a subset where OLMo2-1B was correct in both:
#   1. factual ICL setting
#   2. previous recall/ignore-context result file
#
# Important:
#   This script DOES NOT use the old recall prompt.
#   It DOES NOT parse "original_prompt".
#   The previous recall result file is only used through example_id + correct.
#
# Dataset restriction:
#   keep only country-capital pairs where BOTH country and capital are one word.
#
# This script overwrites:
#   /projectnb/cs505am/students/amao/results/olmo2-1b-instruct-country-capital-both-one-word-both-correct-icl5-results.json

import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

DATA_PATH = "/projectnb/cs505am/students/amao/icl-without-copying/icl_tasks/abstractive/country-capital.json"

ICL_RESULTS_PATH = "/projectnb/cs505am/students/amao/results/olmo2-1b-instruct-country-capital-icl5-results.json"

RECALL_RESULTS_PATH = "/projectnb/cs505am/students/amao/results/olmo2-1b-instruct-recall-icl5-results.json"

OUTPUT_PATH = "/projectnb/cs505am/students/amao/results/olmo2-1b-instruct-country-capital-both-one-word-both-correct-icl5-results.json"


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def normalize_answer(x):
    return x.strip().lower().replace(".", "").replace(",", "")


def load_both_one_word_country_capital_dataset(data_path):
    """
    Load original country-capital pairs.

    Keep only pairs where BOTH:
    - country/input is one word
    - capital/output is one word

    This matches your original filtering rule:
        if not ' ' in item['input'] and not ' ' in item['output']
    """

    dataset = []

    with open(data_path, "r") as f:
        data = json.load(f)

    for item in data:
        country = item["input"].strip()
        capital = item["output"].strip()

        if " " not in country and " " not in capital:
            dataset.append({
                "input": country,
                "output": capital,
            })

    return dataset


def load_correct_countries_from_icl_results(results_path):
    """
    Load correct countries from factual ICL result file.

    Expected result item:
    {
        "example_id": 5,
        "country": "Australia",
        "gold_capital": "Canberra",
        "pred": "Canberra",
        "correct": true
    }
    """

    with open(results_path, "r") as f:
        data = json.load(f)

    correct_countries = set()

    for ex in data["results"]:
        if ex.get("correct", False):
            correct_countries.add(ex["country"].strip())

    return correct_countries


def load_correct_countries_from_previous_recall_results_by_id(
    results_path,
    both_one_word_dataset,
):
    """
    Load correct countries from the previous recall result file.

    This does NOT use or parse the old recall prompt.

    It assumes the previous result has example_id corresponding to the index
    in the filtered both-one-word country-capital dataset.

    Expected result item:
    {
        "example_id": 5,
        "original_prompt": "...",
        "gold": "Canberra",
        "pred": "Canberra",
        "correct": true
    }

    We only use:
    - example_id
    - correct
    """

    with open(results_path, "r") as f:
        data = json.load(f)

    correct_countries = set()

    for ex in data["results"]:
        if not ex.get("correct", False):
            continue

        example_id = ex.get("example_id", None)

        if example_id is None:
            raise ValueError(
                "Previous recall result file does not contain example_id. "
                "Cannot map results back to countries without parsing prompts."
            )

        if example_id < 0 or example_id >= len(both_one_word_dataset):
            raise ValueError(
                f"example_id {example_id} is out of range for filtered dataset "
                f"of size {len(both_one_word_dataset)}."
            )

        country = both_one_word_dataset[example_id]["input"]
        correct_countries.add(country)

    return correct_countries


def build_clean_5shot_prompt(demo_examples, test_country):
    """
    This is the ONLY prompt used for the new model evaluation.

    """

    # prompt = "Find the capital of the country. Only answer with one word.\n\nExamples:\n"
    prompt = "Answer the following country-capital questions. Use the examples to infer the correct answer format. Only output the answer.\n\nExamples:\n\n"

    for ex in demo_examples:
        country = ex["input"].strip()
        capital = ex["output"].strip()
        # prompt += f"{country}: {capital}.\n"
        prompt += f"Example {demo_examples.index(ex) + 1}:\nQuestion: What is the capital of {country}?\nAnswer: {capital}.\n\n"

    prompt += f"{test_country.strip()}:"
    return prompt


def generate_answer(prompt, tokenizer, model, max_new_tokens=10):
    messages = [
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return pred


def build_both_correct_subset(
    data_path,
    icl_results_path,
    previous_recall_results_path,
):
    both_one_word_dataset = load_both_one_word_country_capital_dataset(data_path)

    icl_correct_countries = load_correct_countries_from_icl_results(
        icl_results_path
    )

    previous_recall_correct_countries = (
        load_correct_countries_from_previous_recall_results_by_id(
            previous_recall_results_path,
            both_one_word_dataset,
        )
    )

    both_correct_countries = icl_correct_countries.intersection(
        previous_recall_correct_countries
    )

    subset = [
        ex for ex in both_one_word_dataset
        if ex["input"] in both_correct_countries
    ]

    print(f"Both-one-word dataset size: {len(both_one_word_dataset)}")
    print(f"Correct countries in factual ICL file: {len(icl_correct_countries)}")
    print(f"Correct countries in previous recall file: {len(previous_recall_correct_countries)}")
    print(f"Correct countries in both files: {len(both_correct_countries)}")
    print(f"Final subset size after both-one-word filter: {len(subset)}")

    return subset, both_one_word_dataset


def evaluate_subset(
    tokenizer,
    model,
    model_name,
    data_path,
    icl_results_path,
    previous_recall_results_path,
    out_path,
    num_icl_examples=5,
):
    subset, both_one_word_dataset = build_both_correct_subset(
        data_path=data_path,
        icl_results_path=icl_results_path,
        previous_recall_results_path=previous_recall_results_path,
    )

    assert len(subset) > 0, "Subset is empty. Check paths and result formats."

    assert len(both_one_word_dataset) > num_icl_examples, (
        f"Dataset must contain more than {num_icl_examples} examples."
    )

    demo_examples = both_one_word_dataset[:num_icl_examples]

    results = []
    correct = 0

    for ex_id, ex in enumerate(tqdm(subset)):
        country = ex["input"]
        gold_capital = ex["output"]

        prompt = build_clean_5shot_prompt(
            demo_examples=demo_examples,
            test_country=country,
        )

        prompt += "\n\nNow answer this question:\n"
        prompt += f"Question: What is the capital of {country}? Answer:"
        pred = generate_answer(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=10,
        )

        is_correct = normalize_answer(gold_capital) in normalize_answer(pred)

        if is_correct:
            correct += 1

        results.append({
            "example_id": ex_id,
            "country": country,
            "gold_capital": gold_capital,
            "pred": pred,
            "correct": is_correct,
            "prompt": prompt,
        })

    accuracy = correct / len(subset)

    output = {
        "model": model_name,
        "data_path": data_path,
        "icl_results_path": icl_results_path,
        "previous_recall_results_path": previous_recall_results_path,
        "filter": (
            "both country and capital are one word; "
            "country correct in factual ICL result file; "
            "country correct in previous recall result file by example_id"
        ),
        "task": "clean factual 5-shot country-capital recall on both-correct subset",
        "num_icl_examples": num_icl_examples,
        "icl_examples": demo_examples,
        "num_eval_examples": len(subset),
        "num_correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_path}")
    print(f"Accuracy: {correct}/{len(subset)} = {accuracy:.4f}")

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
    )

    parser.add_argument(
        "--icl_results_path",
        type=str,
        default=ICL_RESULTS_PATH,
    )

    parser.add_argument(
        "--recall_results_path",
        type=str,
        default=RECALL_RESULTS_PATH,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
    )

    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name)

    print("=" * 80)
    print(f"Evaluating {args.model_name}")
    print("Task: clean factual 5-shot country-capital recall")
    print("Model prompt: clean top-5 examples only")
    print("Previous recall file: used only by example_id, not by prompt")
    print("Filter: both country and capital must be one word")
    print(f"Output will overwrite: {args.output_path}")
    print("=" * 80)

    output = evaluate_subset(
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        data_path=args.data_path,
        icl_results_path=args.icl_results_path,
        previous_recall_results_path=args.recall_results_path,
        out_path=args.output_path,
        num_icl_examples=args.num_icl_examples,
    )

    summary = {
        "model": args.model_name,
        "data_path": args.data_path,
        "icl_results_path": args.icl_results_path,
        "previous_recall_results_path": args.recall_results_path,
        "output_path": args.output_path,
        "filter": (
            "both country and capital are one word; "
            "correct in factual ICL and previous recall result files"
        ),
        "task": "clean factual 5-shot country-capital recall on both-correct subset",
        "num_icl_examples": args.num_icl_examples,
        "icl_examples": output["icl_examples"],
        "num_eval_examples": output["num_eval_examples"],
        "num_correct": output["num_correct"],
        "accuracy": output["accuracy"],
    }

    summary_path = args.output_path.replace(
        ".json",
        "knowledge-test-both-correct-summary.json",
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()