import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

DATA_PATH = "/projectnb/cs505am/students/amao/icl-without-copying/icl_tasks/abstractive/country-capital.json"

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


def load_country_capital_dataset(data_path):
    """
    Load country-capital dataset.

    Keep only examples where BOTH:
    - country/input is one word
    - capital/output is one word

    This follows your original filtering rule:
        if not ' ' in item['input'] and not ' ' in item['output']
    """

    city_name_lst = []
    country_name_lst = []

    with open(data_path, "r") as f:
        data = json.load(f)

        for item in data:
            if not " " in item["input"] and not " " in item["output"]:
                city_name_lst.append(item["output"].strip())
                country_name_lst.append(item["input"].strip())

    dataset = []

    for country, capital in zip(country_name_lst, city_name_lst):
        dataset.append({
            "input": country,
            "output": capital,
        })

    return dataset


def build_icl_prompt(demo_examples, test_country):
    prompt_parts = []

    prompt_parts.append(
        "Answer the following country-capital questions. "
        "Given a country name, output only its capital city.\n"
    )

    prompt_parts.append("Examples:\n")

    for i, ex in enumerate(demo_examples):
        country = ex["input"]
        capital = ex["output"]

        prompt_parts.append(f"Example {i + 1}:")
        prompt_parts.append(f"Country: {country}")
        prompt_parts.append(f"Capital: {capital}\n")

    prompt_parts.append("Now answer this question:")
    prompt_parts.append(f"Country: {test_country}")
    prompt_parts.append("Capital:")

    return "\n".join(prompt_parts)


def generate_answer(prompt, tokenizer, model, max_new_tokens=20):
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


def evaluate_dataset(
    data_path,
    out_path,
    tokenizer,
    model,
    model_name,
    num_icl_examples=5,
):
    dataset = load_country_capital_dataset(data_path)

    print(f"Number of examples after both-one-word filtering: {len(dataset)}")

    assert len(dataset) > num_icl_examples, (
        f"Dataset must contain more than {num_icl_examples} examples."
    )

    demo_examples = dataset[:num_icl_examples]
    test_examples = dataset[num_icl_examples:]

    results = []
    correct = 0

    for ex_id, ex in enumerate(tqdm(test_examples)):
        country = ex["input"]
        gold_capital = ex["output"]

        icl_prompt = build_icl_prompt(demo_examples, country)

        pred = generate_answer(
            icl_prompt,
            tokenizer,
            model,
            max_new_tokens=20,
        )

        is_correct = normalize_answer(gold_capital) in normalize_answer(pred)

        if is_correct:
            correct += 1

        results.append({
            "example_id": ex_id + num_icl_examples,
            "country": country,
            "gold_capital": gold_capital,
            "pred": pred,
            "correct": is_correct,
            "icl_prompt": icl_prompt,
        })

    accuracy = correct / len(test_examples)

    output = {
        "model": model_name,
        "data_path": data_path,
        "filter": "both country and capital are one word",
        "num_icl_examples": num_icl_examples,
        "num_total_examples_after_filtering": len(dataset),
        "num_eval_examples": len(test_examples),
        "num_correct": correct,
        "accuracy": accuracy,
        "icl_examples": demo_examples,
        "results": results,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_path}")
    print(f"Accuracy: {correct}/{len(test_examples)} = {accuracy:.4f}")

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
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
    
    )

    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name)

    print("=" * 80)
    print(f"Evaluating {args.model_name} on country-capital dataset")
    print("Filter: both country and capital must be one word")
    print("Using first 5 filtered pairs as ICL examples")
    print("=" * 80)

    output = evaluate_dataset(
        data_path=args.data_path,
        out_path=args.output_path,
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        num_icl_examples=5,
    )

    summary = {
        "model": args.model_name,
        "data_path": args.data_path,
        "output_path": args.output_path,
        "filter": "both country and capital are one word, both correct",
        "num_icl_examples": 5,
        "num_total_examples_after_filtering": output["num_total_examples_after_filtering"],
        "num_eval_examples": output["num_eval_examples"],
        "num_correct": output["num_correct"],
        "accuracy": output["accuracy"],
    }

    summary_path = args.output_path.replace(".json", "knowledge-test-both-correct-summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()