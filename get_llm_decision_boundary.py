import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy import stats
from data_utils import generate_tasks, generate_dataset, generate_context_prompt, generate_context_prompt_reverse, generate_reasoning_prompt, get_hardwired_reasoning_prompt, parse_label


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(2)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")
    
    
def get_model_path(base_model: str = "Llama-3-8B"):
    """Get the model path based on the provided configuration."""
    if base_model == "Llama-3.1-8B":
        path = "meta-llama/Meta-Llama-3.1-8B"
    elif base_model == "Llama-3-8B":
        path = "meta-llama/Meta-Llama-3-8B"
    elif base_model == "Llama-2-7b":
        path = "meta-llama/Llama-2-7b-hf"
    elif base_model == "Llama-2-13b":
        path = "meta-llama/Llama-2-13b-hf"
    else:
        raise ValueError(f"Model currently not supported: {base_model}")
    return path


def load_model_and_tokenizer(base_model: str = "Llama-3-8B", cluster: int = 3, load_bit: int = 8):
    """Load model and tokenizer based on the provided configuration."""
    path = get_model_path(base_model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_in_8bit = load_bit == 8
    load_in_4bit = load_bit == 4
    model = AutoModelForCausalLM.from_pretrained(path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, attn_implementation="eager")
    if load_bit not in [8, 4]:
        model.to(device)
    print(f"Loaded model and tokenizer from {path}")
    return model, tokenizer


def expand_kv_cache(kv_cache, batch_size):
    """
    Expands a KV cache for a single example to a larger batch size.
    
    Args:
    kv_cache (tuple): The original KV cache for a single example.
    batch_size (int): The desired batch size.
    
    Returns:
    tuple: The expanded KV cache.
    """
    # expanded_cache = []
    
    # for layer_cache in kv_cache:
    #     expanded_layer = []
    #     for tensor in layer_cache:
    #         # Repeat the tensor along the batch dimension
    #         expanded_tensor = tensor.repeat(batch_size, 1, 1, 1)
    #         expanded_layer.append(expanded_tensor)
    #     expanded_cache.append(tuple(expanded_layer))
    
    # return tuple(expanded_cache)
    return tuple(
        tuple(tensor.expand(batch_size, -1, -1, -1) for tensor in layer)
        for layer in kv_cache
    )


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_decision_boundary(
    X_train, y_train, xx1, xx2, predictions, model_name="llama3-8b", num_in_context=50, num_in_context_reasoning=0, prompt_format=None, load_bit=None, reverse_inputs=False, save_dir="figures/"
):
    def create_file_name(model_name, num_in_context, save_dir, num_in_context_reasoning=0, prompt_format=None, load_bit=None):
        prefix = f"{model_name}_{num_in_context}incontext"
        if num_in_context_reasoning:
            prefix = f"{prefix}_{num_in_context_reasoning}reasoning"
        if prompt_format:
            prefix = f"{prefix}_{prompt_format}"
        if load_bit:
            prefix = f"{prefix}_{load_bit}bit"
        if reverse_inputs:
            prefix = f"{prefix}_reverse"
        file_name = os.path.join(save_dir, f"{prefix}.png")
        return file_name

    """Plot the decision boundary for the given data and predictions."""
    # Handle NaN values in predictions
    if np.isnan(predictions).any():
        print("Warning: NaN values detected in predictions")
        predictions = np.nan_to_num(predictions, nan=-1)

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Create a color map that includes a color for NaN/-1 values
    cmap = ListedColormap(["#FF9999", "#9999FF", "gray"])
    # Define bounds for each color:
    # - Less than 0.5 (class 0): Red (#FF9999)
    # - Greater than or equal to 0.5 (class 1): Blue (#9999FF)
    # - NaN: Gray
    bounds = [-np.inf, 0.5, np.inf, np.nan]
    norm = BoundaryNorm(bounds, cmap.N)

    contour = ax.contourf(xx1, xx2, predictions, alpha=0.8, cmap=cmap, norm=norm)

    scatter = ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        s=36,
        edgecolor="navy",
        cmap=ListedColormap(["#FF0000", "#0000FF"]),
    )

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.tick_params(axis="both", labelsize=15)

    title = f"{model_name}\n{num_in_context} In-Context Examples, {num_in_context_reasoning}-shot Reasoning"
    if prompt_format:
        title = f"{prompt_format}\n{title}"
    if load_bit:
        title += f"\n{load_bit} bit Quantization"
    
    ax.set_title(title, fontsize=32)
    ax.set_xlabel("Feature 1", fontsize=26)
    ax.set_ylabel("Feature 2", fontsize=26)

    # Add a color bar
    plt.colorbar(contour, ax=ax, label="Prediction")

    # Add a legend for the scatter plot
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper left")
    ax.add_artist(legend1)

    file_name = create_file_name(model_name, num_in_context, save_dir, num_in_context_reasoning, prompt_format, load_bit)
    plt.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)  # Close the figure to free up memory
    return file_name


def create_prompts(args, system_prompt, context_prompt, query_prompt, inputs, reasoning_prompt=None, fixed_label: int=None):
    if "instruct" in args.model_name:
        # Llama instruction prompt format
        prompts = [
            f"### Instructions:\n"
            f"{system_prompt}\n"
            f"### Input:\n"
            f"{context_prompt}\n"
            f"{query_prompt}\n"
            f"Input: {inp}\n"
            f"### Response:\n"
            f"Label: "
            for inp in inputs
        ]

    else:
        if reasoning_prompt:
            prompts = [
                f"{system_prompt}\n{context_prompt}\n{query_prompt}\n{reasoning_prompt}\nInput: {inp}\nSteps: " for inp in inputs
            ]
        else:
            if fixed_label is not None:
                prompts = [f"{system_prompt}\n{context_prompt}\n{query_prompt}\Label: {fixed_label}\nInput: {inp}" for inp in inputs]
            else:
                prompts = [f"{system_prompt}\n{context_prompt}\n{query_prompt}\nInput: {inp}\nLabel: " for inp in inputs]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate a binary classification dataset and split it.")
    parser.add_argument("--model_name", type=str, default="Llama-3-8B", help="Model name")
    parser.add_argument(
        "--num_in_context", type=int, default=128, help="Number of samples per class for in-context learning"
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Grid size for decision boundary plotting")
    parser.add_argument("--num_test_samples", type=int, default=100, help="Number of test examples")
    parser.add_argument("--load_bit", type=int, default=8, help="Bit configuration for loading the model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="01", help="Experiment name")
    parser.add_argument("--num_dimensions", type=int, default=2, help="Number of dimensions for the samples")
    parser.add_argument("--num_tasks", type=int, default=1000, help="Number of tasks to generate")
    parser.add_argument("--num_samples_per_task", type=int, default=800, help="Number of samples per task")  # to sample in-context examples from
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train-test split ratio")
    parser.add_argument(
        "--data_type", type=str, default="linear", help="Type of data to generate, linear, circle or moon"
    )
    parser.add_argument(
        "--class_sep", type=float, default=2.5, help="Class separation for linearly separable data"
    )
    parser.add_argument(
        "--circle_factor", type=float, default=0.5, help="Circle factor for circular data generation"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--num_in_context_reasoning", type=int, default=0, help="Number of in-context examples for reasoning")
    parser.add_argument("--reasoning_set", action="store_true", help="Reasoning examples are hardwired")
    parser.add_argument("--decision_only", action="store_true", help="Only plot the decision boundary")
    parser.add_argument("--plot_save_dir", type=str, default="figures/", help="Directory for saving the plot")
    parser.add_argument("--acc_save_dir", type=str, default="outputs/", help="Directory for saving the plot")
    parser.add_argument("--log_dir", type=str, default="logs/", help="Directory for logging generation results")
    
    parser.add_argument("--reverse_inputs", action="store_true", help="Reverse the input-label order")
    
    args = parser.parse_args()

    set_seed(args.seed)

    class_names_dict = {
        "foobar": ["Foo", "Bar"],
        "01": ["0", "1"],
        "AandB": ["A", "B"],
        "reverse_foobar": ["Bar", "Foo"],
        "PosNeg": ["Positive", "Negative"],
        "yesno": ["Yes", "No"],
    }

    if args.exp_name not in class_names_dict:
        raise ValueError(f"Unknown experiment name: {args.exp_name}")

    class_names = class_names_dict[args.exp_name]

    # Generate tasks and split into training and testing sets
    dataset_x, dataset_y = generate_tasks(
        num_tasks=args.num_tasks,
        num_samples_per_task=args.num_samples_per_task,
        num_dimensions=args.num_dimensions,
        seed=args.seed,
        data_type=args.data_type,
        class_sep=args.class_sep,
        factor=args.circle_factor,
    )
    # meta_train_X, meta_test_X, meta_train_y, meta_test_y = train_test_split(  # maybe used for training the model later?
    #     dataset_x, dataset_y, train_size=args.train_ratio  # currently randomly splits; not train-test sequential split
    # )
    meta_train_X, meta_train_y = dataset_x, dataset_y  # use all data for training

    
    # def plot_scatter(x, y, labels):
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(x[labels == 0], y[labels == 0], c='red', label='Label 0')
    #     plt.scatter(x[labels == 1], y[labels == 1], c='blue', label='Label 1')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title(f'Scatter Plot of Input Data (Seed {args.seed})')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f"{args.plot_save_dir}/seed{args.seed}_task0.png", dpi=300)
        
    
    print(f"Meta_train_X shape: {meta_train_X.shape}")
    # print(f"Meta_test_X shape: {meta_test_X.shape}")
    print(f"Meta_train_y shape: {meta_train_y.shape}")
    # plot_scatter(meta_train_X[0, :, 0], meta_train_X[0, :, 1], meta_train_y[0])
    print("-" * 50)

    if args.reverse_inputs:
        system_prompt = f"Given labels and pairs of numbers for the labels, predict the input pair of numbers for a new label based on the provided data. Try to get the best estimate of what the pair of numbers would be. Answer with only positive integers."
        query_prompt = "What is the input for this label?"
    else:
        system_prompt = f"Given pairs of numbers and their labels, predict the label for a new input pair of numbers based on the provided data. Answer with only one of the labels '{class_names[0]}' and '{class_names[1]}'."
        query_prompt = "What is the label for this input?"

    context_x, context_y, query_x, query_y, reasoning_x, reasoning_y = generate_dataset(args, meta_train_X, meta_train_y)

    print("-" * 10, "Context X, Y shapes:", context_x.shape, context_y.shape, query_x.shape, query_y.shape)
    print("-" * 10, "Reasoning X, Y shapes:", reasoning_x.shape, reasoning_y.shape)

    model, tokenizer = load_model_and_tokenizer(base_model=args.model_name, load_bit=args.load_bit)
    # Tokenization robustness. This handles cases where the tokenizer may split "Foo" differently when preceded by a space
    tokens = class_names
    token_ids_foo = tokenizer(tokens[0], return_tensors="pt")["input_ids"][0][-1]
    token_ids_bar = tokenizer(tokens[1], return_tensors="pt")["input_ids"][0][-1]
    token_ids_foo1 = tokenizer(f" {tokens[0]}", return_tensors="pt")["input_ids"][0][-1]
    token_ids_bar1 = tokenizer(f" {tokens[1]}", return_tensors="pt")["input_ids"][0][-1]
    
    # desired_task_idx = [0]  # only on the first task for demo
    desired_task_idx = range(len(meta_train_X))  # all tasks

    accuracies = []
    samples_decision = []
    samples_test = []
    
    max_new_tokens = 128 if args.reasoning_set else 1
    min_new_tokens = 20 if args.reasoning_set else 1
    
    for task_idx in desired_task_idx:
        
        task_context_x = context_x[task_idx]
        task_context_y = context_y[task_idx]
        
        task_reasoning_x = reasoning_x[task_idx]
        task_reasoning_y = reasoning_y[task_idx]

        x1_min, x1_max = task_context_x[:, 0].min() - 1, task_context_x[:, 0].max() + 1
        x2_min, x2_max = task_context_x[:, 1].min() - 1, task_context_x[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(
            np.linspace(x1_min, x1_max, args.grid_size), np.linspace(x2_min, x2_max, args.grid_size)
        )
        xx1_flat, xx2_flat = xx1.ravel(), xx2.ravel()
        inputs = [f"{int(x)} {int(y)}" for x, y in zip(xx1_flat, xx2_flat)]  # actual x, y coordinates

        if args.reverse_inputs:
            context_prompt = generate_context_prompt_reverse(X=task_context_x, y=task_context_y, class_names=class_names)
        else:
            context_prompt = generate_context_prompt(X=task_context_x, y=task_context_y, class_names=class_names)
        if args.num_in_context_reasoning > 0:
            if args.reasoning_set:
                reasoning_prompt = get_hardwired_reasoning_prompt()
            else:
                reasoning_prompt = generate_reasoning_prompt(X=task_reasoning_x, y=task_reasoning_y, class_names=class_names)
                print(f"Context prompt in {task_idx}'th task:\n{context_prompt}")
                print(f"Reasoning prompt in {task_idx}'th task:\n{reasoning_prompt}")
                break
        else:
            reasoning_prompt = None
        
        if args.reverse_inputs:
            prompts_0 = create_prompts(args, system_prompt, context_prompt, query_prompt, inputs, reasoning_prompt, fixed_label=0)
            prompts_1 = create_prompts(args, system_prompt, context_prompt, query_prompt, inputs, reasoning_prompt, fixed_label=1)
            print(f"Prompt sample in {task_idx}'th task:\n{prompts_0[0]}")
            print(f"Prompt sample in {task_idx}'th task:\n{prompts_1[0]}")
        else:
            prompts = create_prompts(args, system_prompt, context_prompt, query_prompt, inputs, reasoning_prompt)
            print(f"Prompt sample in {task_idx}'th task:\n{prompts[0]}")

        # Store the KV cache for the in-context examples to speed up.
        inputs_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False)["input_ids"]
        max_input_len = inputs_ids.shape[1] + 10  # 10 is a random Buffer for tokenization differences

        predictions = np.zeros(xx1_flat.shape[0])
        logits_pred = np.zeros((xx1_flat.shape[0], 2))

        prompt_input_ids = tokenizer(prompts[0] if not args.reverse_inputs else prompts_0[0], return_tensors="pt", padding=True, truncation=False)[
            "input_ids"
        ]
        in_context_ids = prompt_input_ids[:, :-max_input_len].to(model.device)

        with torch.inference_mode():
            in_context_kv_cache = model(input_ids=in_context_ids, return_dict=True).past_key_values
        
        
        # Decision boundary plotting
        if task_idx == 0:  # only plot the decision boundary for the first task
            if args.reverse_inputs:
                def calculate_input_logprobs(prompts):
                    log_probs = []
                    for i in tqdm(range(0, len(prompts), args.batch_size)):
                        batch_prompts = prompts[i : i + args.batch_size]
                        
                        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
                        input_ids = inputs["input_ids"].to(model.device)
                        attention_mask = inputs["attention_mask"]
                        
                        # Split the input_ids into context and question parts
                        context_length = in_context_kv_cache[0][0].shape[2]  # Get the length of the cached context
                        
                        # Adjust input_ids and attention_mask to only include the new tokens
                        input_ids = input_ids[:, context_length:]
                        attention_mask = attention_mask[:, context_length:]
                        
                        # Ensure that we're only processing the last few tokens if the input is too long
                        max_new_tokens = 10  # Adjust this value as needed
                        if input_ids.shape[1] > max_new_tokens:
                            input_ids = input_ids[:, -max_new_tokens:]
                            attention_mask = attention_mask[:, -max_new_tokens:]
                            
                        # Expand the in_context_kv_cache to match the batch size
                        expanded_kv_cache = expand_kv_cache(in_context_kv_cache, args.batch_size)

                        with torch.inference_mode():
                            # Use the expanded KV cache to get the outputs for the question part
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                past_key_values=expanded_kv_cache,
                                use_cache=False,  # Set this to False to avoid issues with mismatched sizes
                                return_dict=True
                            )
                        
                        # Get the logits for the last three tokens (corresponding to the input pair and space)
                        last_token_logits = outputs.logits[:, -3:, :]
                        
                        # Calculate log probabilities
                        log_probs_batch = F.log_softmax(last_token_logits, dim=-1)
                        
                        # Get the log probs for the actual tokens
                        actual_log_probs = log_probs_batch.gather(2, input_ids[:, -3:].unsqueeze(-1)).squeeze(-1)
                        
                        # Sum the log probs for each input pair, excluding the space
                        # Assuming the space is always in the middle, we sum the first and last log prob
                        pair_log_probs = actual_log_probs[:, 0] + actual_log_probs[:, 2]
                        
                        # Sum the log probs for each input pair
                        log_probs.extend(pair_log_probs.tolist())
                    
                    return log_probs

                log_probs_0 = calculate_input_logprobs(prompts_0)
                log_probs_1 = calculate_input_logprobs(prompts_1)

                # Compare log probabilities and make predictions
                predictions = []
                logits_pred = []
                for lp0, lp1 in zip(log_probs_0, log_probs_1):
                    if lp0 > lp1:
                        predictions.append(0)
                        logits_pred.append([lp0, lp1])
                    else:
                        predictions.append(1)
                        logits_pred.append([lp1, lp0])

                predictions = np.array(predictions)
                logits_pred = np.array(logits_pred)
            else:
                for i in tqdm(range(0, len(prompts), args.batch_size)):
                    batch_prompts = prompts[i : i + args.batch_size]
                    batch_size = len(batch_prompts)  # batch size may not be equal to args.batch_size for the last batch

                    total_prompt = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)[
                        "input_ids"
                    ].to(model.device)
                    prompt_length = total_prompt.shape[1]
                    this_in_context_ids = total_prompt[:, : len(in_context_ids[0])]
                    question_ids = total_prompt[:, len(in_context_ids[0]) :]
                    
                    assert torch.equal(this_in_context_ids[0], in_context_ids[0])
                    
                    in_context_kv_cache_expanded = expand_kv_cache(in_context_kv_cache, batch_size)
                    with torch.inference_mode():
                        in_context_q_kv_cache = model(
                            question_ids[:, :-1], past_key_values=in_context_kv_cache_expanded, return_dict=True
                        ).past_key_values

                        generations = model.generate(
                            input_ids=total_prompt,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            past_key_values=in_context_q_kv_cache,
                            pad_token_id=tokenizer.eos_token_id,
                            output_scores=True,
                            return_dict_in_generate=True,
                            output_attentions=True,
                        )
                        logits = generations["scores"][-1]
                    if not args.reasoning_set:
                        logit_bar = max(logits[0, token_ids_bar].item(), logits[0, token_ids_bar1].item())
                        logit_foo = max(logits[0, token_ids_foo].item(), logits[0, token_ids_foo1].item())
                        generated_texts = tokenizer.batch_decode(
                            generations["sequences"][:, -1:], skip_special_tokens=True  # only check last token
                        )
                    else:
                        generated_sequences = generations["sequences"][:, prompt_length:]
                        generated_texts = tokenizer.batch_decode(
                            generated_sequences, skip_special_tokens=True
                        )
                    for idx, (generated_text, x_val, y_val) in enumerate(
                        zip(generated_texts, xx1_flat[i : i + batch_size], xx2_flat[i : i + batch_size])
                    ):
                        print(f"idx: {i + idx}, x: {x_val}, y: {y_val}")
                        print(f"Generated text: {generated_text}")
                        if args.reasoning_set:
                            label = parse_label(generated_text)
                            predictions[i + idx] = label
                            samples_decision.append({
                                "task_idx": task_idx,
                                "idx": i + idx,
                                "inputs": f"{int(x_val)} {int(y_val)}",
                                "generated_text": generated_text,
                                "prediction": label
                            })
                        # Check if the generated text contains the class names, if not, use the logit to predict
                        else:
                            if class_names[0].lower() in generated_text.lower():
                                predictions[i + idx] = 0
                            elif class_names[1].lower() in generated_text.lower():
                                predictions[i + idx] = 1
                            else:
                                if logit_bar > logit_foo:
                                    predictions[i + idx] = 1
                                    logits_pred[i + idx] = [logit_bar, logit_foo]
                                else:
                                    predictions[i + idx] = 0
                                    logits_pred[i + idx] = [logit_foo, logit_bar]

            llm_predictions = predictions.reshape(xx1.shape)
            file_name = plot_decision_boundary(
                task_context_x,
                task_context_y,
                xx1,
                xx2,
                llm_predictions,
                model_name=args.model_name,
                num_in_context=args.num_in_context,
                num_in_context_reasoning=args.num_in_context_reasoning,
                prompt_format=args.exp_name,
                load_bit=args.load_bit,
                reverse_inputs=args.reverse_inputs,
                save_dir=args.plot_save_dir,
            )
            print(f"Decision boundary plot saved as {file_name}")
            
        file_name_logs = os.path.join(args.log_dir, os.path.basename(file_name)).replace(".png", ".json")
        with open(file_name_logs, "w") as f:
            json.dump(samples_decision, f, indent=4)
            
        if args.decision_only:
            break
        
        # Test set evaluation
        task_query_x = query_x[task_idx]
        task_query_y = query_y[task_idx]
        
        test_inputs = [f"{int(x)} {int(y)}" for x, y in zip(task_query_x[:, 0], task_query_x[:, 1])]  # x is a pair of numbers in 2-dimensional space
        test_prompts = create_prompts(args, system_prompt, context_prompt, query_prompt, test_inputs)
        
        predictions_test = np.zeros(len(test_inputs))
        logits_pred_test = np.zeros((len(test_inputs), 2))
        
        for i in tqdm(range(0, len(test_prompts))):
            batch_prompts = test_prompts[i : i + 1]
        
            total_prompt = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)[
                "input_ids"
            ].to(model.device)
            this_in_context_ids = total_prompt[:, : len(in_context_ids[0])]
            question_ids = total_prompt[:, len(in_context_ids[0]) :]

            assert torch.equal(this_in_context_ids, in_context_ids)
            with torch.inference_mode():
                in_context_q_kv_cache = model(
                    question_ids[:, :-1], past_key_values=in_context_kv_cache, return_dict=True
                ).past_key_values

                generations = model.generate(
                    input_ids=total_prompt,
                    do_sample=False,
                    max_new_tokens=1,
                    past_key_values=in_context_q_kv_cache,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )
                logits = generations["scores"][-1]
            
            logit_bar = max(logits[0, token_ids_bar].item(), logits[0, token_ids_bar1].item())
            logit_foo = max(logits[0, token_ids_foo].item(), logits[0, token_ids_foo1].item())
            generated_texts = tokenizer.batch_decode(
                generations["sequences"][:, -1:], skip_special_tokens=True
            )
            for idx, generated_text in enumerate(generated_texts):  # batched generation
                # Check if the generated text contains the class names, if not, use the logit to predict
                if class_names[0].lower() in generated_text.lower():
                    predictions_test[i + idx] = 0
                elif class_names[1].lower() in generated_text.lower():
                    predictions_test[i + idx] = 1
                else:
                    if logit_bar > logit_foo:
                        predictions_test[i + idx] = 1
                        logits_pred_test[i + idx] = [logit_bar, logit_foo]
                    else:
                        predictions_test[i + idx] = 0
                        logits_pred_test[i + idx] = [logit_foo, logit_bar]
                samples_test.append({
                    "task_idx": task_idx,
                    "idx": i + idx,
                    "num_in_context": args.num_in_context,
                    "prompt": test_prompts[i + idx],
                    "generated_text": generated_text,
                    "prediction": predictions_test[i + idx],
                    "true_label": task_query_y[i + idx]
                })
        # print(f"Task {task_idx} Test predictions ({np.array(predictions_test).shape}): {predictions_test}")
        # print(f"Task {task_idx} Test labels ({task_query_y.shape}): {task_query_y}")
        test_accuracy = np.mean(np.array(predictions_test) == task_query_y)
        print(f"Task {task_idx} Test accuracy: {test_accuracy}")
        
        accuracies.append(test_accuracy)
    
    if not args.decision_only:
        mean_accuracy = np.mean(accuracies)
        std_error = stats.sem(accuracies)
        
        print(f"Mean test accuracy: {mean_accuracy:.4f} ± {std_error:.4f}")
        
        file_name_json = os.path.join(args.acc_save_dir, os.path.basename(file_name)).replace(".png", ".json")
        config = {
            "model_name": args.model_name,
            "num_in_context": args.num_in_context,
            "grid_size": args.grid_size,
            "num_test_samples": args.num_test_samples,
            "load_bit": args.load_bit,
            "seed": args.seed,
            "exp_name": args.exp_name,
            "num_dimensions": args.num_dimensions,
            "num_tasks": args.num_tasks,
            "num_samples_per_task": args.num_samples_per_task,
            "train_ratio": args.train_ratio,
            "data_type": args.data_type,
            "class_sep": args.class_sep,
            "circle_factor": args.circle_factor,
            "plot_save_dir": args.plot_save_dir,
            "acc_save_dir": args.acc_save_dir,
            "plot_save_file": file_name,
            "acc_save_file": file_name_json
        }
        with open(file_name_json, "w") as f:
            json.dump({
                "mean_accuracy": mean_accuracy, 
                "std_error": std_error, 
                "accuracies": accuracies, 
                "config": config
            }, f, indent=4)
        with open(os.path.join("inputs/", os.path.basename(file_name_json)), "w") as f:
            json.dump(samples_test, f, indent=4)

if __name__ == "__main__":
    main()
