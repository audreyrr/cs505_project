import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

MODEL_NAME = ["allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-0425-1B-Instruct", "allenai/Olmo-3-7B-Instruct"]

RECALL_DATA_PATH = "/projectnb/cs505am/students/amao/generated_datasets/country-capital-cc-pair-recall.json"
IN_CXT_DATA_PATH = "/projectnb/cs505am/students/amao/generated_datasets/country-capital-cc-pair-in-cxt.json"

OUT_RECALL_PATH = "/projectnb/cs505am/students/amao/results/olmo2-7b-instruct-recall-dataset-results.json"
OUT_IN_CXT_PATH = "/projectnb/cs505am/students/amao/results/olmo2-7b-instruct-in-cxt-dataset-results.json"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    model.eval()
    return tokenizer, model


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


def normalize_answer(x):
    return x.strip().lower().replace(".", "").replace(",", "")


def evaluate_dataset(data_path, out_path, tokenizer, model):
    with open(data_path, "r") as f:
        dataset = json.load(f)

    results = []
    correct = 0

    for ex in tqdm(dataset):
        prompt = ex["prompt"]
        gold = ex["answer"]

        pred = generate_answer(prompt, tokenizer, model)

        is_correct = normalize_answer(gold) in normalize_answer(pred)

        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
        })

    acc = correct / len(dataset)

    output = {
        "model": MODEL_NAME,
        "data_path": data_path,
        "accuracy": acc,
        "num_examples": len(dataset),
        "num_correct": correct,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {out_path}")
    print(f"Accuracy: {correct}/{len(dataset)} = {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--recall_data_path", type=str, default=RECALL_DATA_PATH)
    parser.add_argument("--in_cxt_data_path", type=str, default=IN_CXT_DATA_PATH)
    parser.add_argument("--out_recall_path", type=str, default=OUT_RECALL_PATH)
    parser.add_argument("--out_in_cxt_path", type=str, default=OUT_IN_CXT_PATH)

    args = parser.parse_args()

    tokenizer, model = load_model()

    print("Evaluating recall dataset...")
    evaluate_dataset(RECALL_DATA_PATH, OUT_RECALL_PATH, tokenizer, model)

    print("Evaluating in-context dataset...")
    evaluate_dataset(IN_CXT_DATA_PATH, OUT_IN_CXT_PATH, tokenizer, model)


if __name__ == "__main__":
    main()

# recall dataset:Accuracy: 131/143 = 0.9161
# in-context dataset:Accuracy: 130/143 = 0.9091