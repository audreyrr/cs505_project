import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

RECALL_DATA_PATH = "/projectnb/cs505am/students/amao/generated_datasets/country-capital-cc-pair-recall.json"
IN_CXT_DATA_PATH = "/projectnb/cs505am/students/amao/generated_datasets/country-capital-cc-pair-in-cxt.json"

OUT_RECALL_PATH = "/projectnb/cs505am/students/amao/results/olmo2-1b-instruct-recall-icl5-results.json"
OUT_IN_CXT_PATH = "/projectnb/cs505am/students/amao/results/olmo2-1b-instruct-in-cxt-icl5-results.json"


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


def build_icl_prompt(demo_examples, test_prompt):
    """
    Build an in-context learning prompt using the first 5 examples.

    Expected dataset format:
    {
        "prompt": "...",
        "answer": "..."
    }
    """

    prompt_parts = []

    prompt_parts.append(
        "Answer the following country-capital questions. "
        "Use the examples to infer the correct answer format. "
        "Only output the answer.\n"
    )

    prompt_parts.append("Examples:\n")

    for i, ex in enumerate(demo_examples):
        demo_prompt = ex["prompt"].strip()
        demo_answer = ex["answer"].strip()

        prompt_parts.append(f"Example {i + 1}:")
        prompt_parts.append(f"Question: {demo_prompt}")
        prompt_parts.append(f"Answer: {demo_answer}\n")

    prompt_parts.append("Now answer this question:")
    prompt_parts.append(f"Question: {test_prompt.strip()}")
    prompt_parts.append("Answer:")

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
    with open(data_path, "r") as f:
        dataset = json.load(f)

    assert len(dataset) > num_icl_examples, (
        f"Dataset must contain more than {num_icl_examples} examples."
    )

    demo_examples = dataset[:num_icl_examples]
    test_examples = dataset[num_icl_examples:]

    results = []
    correct = 0

    for ex_id, ex in enumerate(tqdm(test_examples)):
        original_prompt = ex["prompt"]
        gold = ex["answer"]

        icl_prompt = build_icl_prompt(demo_examples, original_prompt)

        pred = generate_answer(
            icl_prompt,
            tokenizer,
            model,
            max_new_tokens=20,
        )

        is_correct = normalize_answer(gold) in normalize_answer(pred)

        if is_correct:
            correct += 1

        results.append({
            "example_id": ex_id + num_icl_examples,
            "original_prompt": original_prompt,
            "icl_prompt": icl_prompt,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
        })

    accuracy = correct / len(test_examples)

    output = {
        "model": model_name,
        "data_path": data_path,
        "num_icl_examples": num_icl_examples,
        "num_total_examples": len(dataset),
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
        "--recall_data_path",
        type=str,
        default=RECALL_DATA_PATH,
    )

    parser.add_argument(
        "--in_cxt_data_path",
        type=str,
        default=IN_CXT_DATA_PATH,
    )

    parser.add_argument(
        "--out_recall_path",
        type=str,
        default=OUT_RECALL_PATH,
    )

    parser.add_argument(
        "--out_in_cxt_path",
        type=str,
        default=OUT_IN_CXT_PATH,
    )

    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name)

    print("=" * 80)
    print("Evaluating recall dataset with 5-shot ICL...")
    print("=" * 80)

    recall_output = evaluate_dataset(
        data_path=args.recall_data_path,
        out_path=args.out_recall_path,
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        num_icl_examples=args.num_icl_examples,
    )

    print("=" * 80)
    print("Evaluating in-context dataset with 5-shot ICL...")
    print("=" * 80)

    in_cxt_output = evaluate_dataset(
        data_path=args.in_cxt_data_path,
        out_path=args.out_in_cxt_path,
        tokenizer=tokenizer,
        model=model,
        model_name=args.model_name,
        num_icl_examples=args.num_icl_examples,
    )

    summary = {
        "model": args.model_name,
        "num_icl_examples": args.num_icl_examples,
        "recall_dataset": {
            "data_path": args.recall_data_path,
            "out_path": args.out_recall_path,
            "num_eval_examples": recall_output["num_eval_examples"],
            "num_correct": recall_output["num_correct"],
            "accuracy": recall_output["accuracy"],
        },
        "in_context_dataset": {
            "data_path": args.in_cxt_data_path,
            "out_path": args.out_in_cxt_path,
            "num_eval_examples": in_cxt_output["num_eval_examples"],
            "num_correct": in_cxt_output["num_correct"],
            "accuracy": in_cxt_output["accuracy"],
        },
    }

    summary_path = "/projectnb/cs505am/students/amao/results/olmo2-1b-instruct-icl5-summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()