import pandas as pd
import ast
import random
from pathlib import Path
from scipy import stats
import numpy as np


class BilingualFewShotDataset:
    def __init__(
        self,
        dataset_path: Path,
        lang1: str,
        lang2: str,
        n_shots: int = 10,
        random_pairs: bool = False,
        lang_labels: bool = True,
        seed: int = 1234,
    ):
        self.lang1, self.lang2, self.n_shots = lang1, lang2, n_shots
        self.seed = seed
        self.labels = {
            "eng": "English",
            "spa": "Español",
            "jpn": "日本語",
            "fra": "Français",
            "por": "Português",
            "ita": "Italiano",
            "pol": "Polski",
            "cmn": "中文",
            "ind": "Bahasa Indonesia",
            "arb": "العربية",
            "swe": "Svenska",
        }
        self.lang_labels = lang_labels
        non_text_cols = ["synset_id", "definition"]
        text_columns = pd.read_csv(dataset_path, nrows=0).columns.difference(
            non_text_cols
        )
        self.dataframe = pd.read_csv(
            dataset_path, converters={col: ast.literal_eval for col in text_columns}
        )
        if random_pairs:
            self.prompts, self.targets = self._build_few_shot_random_pairs()
        else:
            self.prompts, self.targets = self._build_few_shot_pairs()

    def __len__(self):
        return len(self.prompts)

    def _build_few_shot_pairs(self):
        prompts_list, targets_list = [], []
        if self.lang_labels:
            label1 = f"{self.labels[self.lang1]}: "
            label2 = f"{self.labels[self.lang2]}: "
        else:
            label1 = ""
            label2 = ""
        for idx, row in self.dataframe.iterrows():
            few_shot_examples = self.dataframe.drop(idx).sample(
                self.n_shots, random_state=self.seed + idx
            )
            prompt = "\n".join(
                f"{label1}'{ex[self.lang1][0].replace('_', ' ')}' – {label2}'{ex[self.lang2][0].replace('_', ' ')}'"
                for _, ex in few_shot_examples.iterrows()
            )
            prompt += f"\n{label1}'{row[self.lang1][0].replace('_', ' ')}' – {label2}'"
            prompts_list.append(prompt)
            if self.lang2 == "eng":
                targets_list.append([row[self.lang2][0].replace('_', ' ')])
            else:
                targets_list.append(row[self.lang2][0].replace('_', ' '))
        return prompts_list, targets_list

    def _build_few_shot_random_pairs(self):
        def scramble_row(s):
            scrambled = scramble(s)
            return f"'{scrambled}' - '{scrambled}'"

        def scramble(s):
            return "".join(random.sample(s, len(s)))

        prompts_list, targets_list = [], []
        label1 = self.labels[self.lang1]
        label2 = self.labels[self.lang2]
        for idx, row in self.dataframe.iterrows():
            few_shot_examples = self.dataframe.drop(idx).sample(self.n_shots)
            prompt = "\n".join(
                scramble_row(ex[self.lang1][0])
                for _, ex in few_shot_examples.iterrows()
            )
            scrambled_target = scramble(row[self.lang1][0])
            prompt += f"\n{scrambled_target} – '"
            prompts_list.append(prompt)
            if self.lang2 == "eng":
                targets_list.append([scrambled_target])
            else:
                targets_list.append(row[self.lang2])
        return prompts_list, targets_list


def compute_prompt_lengths(batch_prompts, full_texts, tokenizer):
    prompt_lengths = []
    for prompt, full_text in zip(batch_prompts, full_texts):
        prompt_length = len(prompt)
        prompt_token_length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        necessary_prompt_token_length = prompt_token_length
        tokenized_prompt = tokenizer(full_text, add_special_tokens=False)
        tokenized_prompt_beginning = tokenized_prompt.input_ids[
            :necessary_prompt_token_length
        ]
        tokenized_prompt_beginning_length = len(
            tokenizer.decode(tokenized_prompt_beginning)
        )
        if tokenized_prompt_beginning_length == prompt_length:
            prompt_lengths.append(necessary_prompt_token_length)
        else:
            if tokenized_prompt_beginning_length > prompt_length:
                while tokenized_prompt_beginning_length > prompt_length:
                    necessary_prompt_token_length = necessary_prompt_token_length - 1
                    tokenized_prompt_beginning = tokenized_prompt.input_ids[
                        :necessary_prompt_token_length
                    ]
                    tokenized_prompt_beginning_length = len(
                        tokenizer.decode(tokenized_prompt_beginning)
                    )
            elif tokenized_prompt_beginning_length < prompt_length:
                while tokenized_prompt_beginning_length <= prompt_length:
                    necessary_prompt_token_length = necessary_prompt_token_length + 1
                    tokenized_prompt_beginning = tokenized_prompt.input_ids[
                        :necessary_prompt_token_length
                    ]
                    tokenized_prompt_beginning_length = len(
                        tokenizer.decode(tokenized_prompt_beginning)
                    )
                necessary_prompt_token_length = necessary_prompt_token_length - 1
            prompt_lengths.append(necessary_prompt_token_length)
    return prompt_lengths


def evaluate_translation_accuracy(
    model,
    tokenizer,
    dataset,
    device,
    batch_size=32,
    max_new_tokens=50,
    random_pairs=False,
):
    num_correct = 0
    num_total = 0
    predictions = []
    for i in range(0, len(dataset), batch_size):
        batch_prompts = dataset.prompts[i : i + batch_size]
        batch_targets = dataset.targets[i : i + batch_size]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        input_ids, attention_mask = enc.input_ids, enc.attention_mask
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        for idx, seq in enumerate(gen_ids):
            prompt_len = (attention_mask[idx] == 1).sum().item()
            pad_beginning = (attention_mask[idx] == 0).sum().item()
            gen_text = tokenizer.decode(
                seq[prompt_len + pad_beginning :], skip_special_tokens=True
            ).strip()
            tgt_texts = [t.strip() for t in batch_targets[idx]]

            if any(gen_text.startswith(t + "'") for t in tgt_texts):
                num_correct += 1
                predictions.append(True)
            else:
                predictions.append(False)

            num_total += 1

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return accuracy, predictions


def evaluate_translation_accuracy_with_ci(
    model,
    tokenizer,
    dataset,
    device,
    batch_size=32,
    max_new_tokens=50,
    random_pairs=False,
    confidence_level=0.95,
):
    accuracy, predictions = evaluate_translation_accuracy(
        model, tokenizer, dataset, device, batch_size, max_new_tokens, random_pairs
    )

    n_samples = len(dataset)
    z_value = stats.norm.ppf((1 + confidence_level) / 2)

    std_error = np.sqrt((accuracy * (1 - accuracy)) / n_samples)

    ci_length = z_value * std_error
    ci_lower = max(0, accuracy - ci_length)
    ci_upper = min(1, accuracy + ci_length)

    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_samples": n_samples,
    }, predictions
