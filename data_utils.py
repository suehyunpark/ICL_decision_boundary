import re
from sklearn.datasets import make_circles, make_moons
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def generate_x_y(
    num_samples,
    num_dimensions,
    seed,
    data_type="linear",
    factor=0.5,
    class_sep=1,
    noise_moon=0.05,
    num_classes=2,
):
    """Generate X and y data based on the specified data type."""
    if data_type == "linear":
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_dimensions,
            n_informative=num_dimensions,
            n_redundant=0,  # no redundant features
            n_clusters_per_class=1,  # each class is a single cluster
            flip_y=0,  # no noise
            shuffle=True,
            random_state=seed,
            n_classes=num_classes,
            class_sep=class_sep,  # make classes clearly separable
        )
    elif data_type == "circle":
        X, y = make_circles(n_samples=num_samples, shuffle=True, noise=0.05, random_state=seed, factor=factor)
    elif data_type == "moon":
        X, y = make_moons(n_samples=num_samples, shuffle=True, noise=noise_moon, random_state=seed)

    # Normalize X to [0, 1] and then scale to [0, 100]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 100 * (X - X_min) / (X_max - X_min)

    return X, y


def generate_tasks(
    num_tasks, num_samples_per_task, num_dimensions, seed, data_type="linear", factor=0.5, class_sep=2
):
    """Generate multiple tasks, each with its own dataset."""
    # Create empty arrays to store X and y data
    X_data = np.zeros((num_tasks, num_samples_per_task, num_dimensions))
    Y_data = np.zeros((num_tasks, num_samples_per_task))

    for i in range(num_tasks):
        X, y = generate_x_y(
            num_samples=num_samples_per_task,
            num_dimensions=num_dimensions,
            seed=seed + i,
            data_type=data_type,
            factor=factor,
            class_sep=class_sep,
        )
        X_data[i] = X
        Y_data[i] = y

    print(f"Generated {num_tasks} tasks with {num_samples_per_task} samples each.")
    return X_data, Y_data


def generate_context_prompt(X, y, class_names):
    y_named = [class_names[int(label)] for label in y]

    prompt = ""
    for features, label in zip(X, y_named):
        features_str = " ".join(f"{int(num)}" for num in np.round(features))
        prompt += f"Input: {features_str}\nLabel: {label}\n"
    return prompt


def generate_reasoning_prompt(X, y, class_names):
    y_named = [class_names[int(label)] for label in y]

    prompt = ""
    for features, label in zip(X, y_named):
        features_str = " ".join(f"{int(num)}" for num in np.round(features))
        prompt += f"Input: {features_str}\nLabel: {label}\n"
    return prompt


def get_hardwired_reasoning_prompt(num_in_context_reasoning=4):
    prompt = """Input: 20 44
Steps: A possible pattern is that label 0 seems to be assigned when the first number is smaller than the second number, or when both numbers are relatively low, while label 1 seems to be assigned when the first number is larger than the second number, or when both numbers are relatively high. For the input (20,44), observe that the first number (20) is smaller than the second number (44), and both numbers are relatively low compared to most of the other inputs. This pattern is more similar to the inputs labeled 0.
Label: 0
Input: 58 32
Steps: Let's analyze the given input (58, 32) based on the previously provided data. By examining the existing pairs, we can observe a potential trend: inputs where the first number notably exceeds the second tend to be labeled as 1. In our case, 58 is significantly larger than 32. Looking at similar examples, we see (66, 30) and (72, 51) both labeled as 1, which aligns with our current input's pattern. While some label 0 cases like (25, 52) and (22, 61) show the opposite trend, our input more closely resembles the label 1 instances. The overall magnitude of the numbers also appears to play a role, with higher values often associated with label 1. Given these observations and the apparent similarity to previous label 1 cases, we can reasonably infer that the most likely label for the input (58, 32) is 1.
Label: 1
Input: 73 54
Steps: First examine the relationship between the given pairs of numbers and their corresponding labels. The label seems to depend on the values of the two numbers, but not in a simple linear fashion. We can notice that pairs where both numbers are relatively close or both high tend to have the label '1', while pairs with greater differences or smaller values tend to have the label '0'. For the pair '73 54', the numbers are relatively close but not as close as other '1' labeled pairs. However, both numbers are high, similar to other pairs with label '1'. Thus, based on this pattern, we predict the label as '1'.
Label: 1
Input: 27 39
Steps: Start by analyzing the patterns in the given pairs and their labels. Noticing that pairs with smaller numbers or greater differences between them often have the label '0', while pairs with higher or more similar values usually have the label '1'. The input "27 39" consists of relatively smaller numbers with a moderate difference between them, similar to pairs like "25 52" and "29 54," which both have the label '0'. Therefore, based on this pattern of smaller numbers or greater differences, we predict the label as '0'.
Label: 0"""
    if num_in_context_reasoning == 8:
        prompt = f"""{prompt}
Input: 56 59
Steps: Let's analyze the input (56, 59) based on the patterns observed in the given data. First, we notice that these numbers are relatively close to each other, with only a small difference of 3. Looking at previous examples, we see that pairs with similar values often tend to be labeled as 1. However, we should also consider the overall magnitude of the numbers. In this case, both 56 and 59 fall in the mid-range of the values we've seen, not particularly high or low. Comparing to other examples, we find that (67, 65) was labeled 1, which is quite similar to our current input in terms of both closeness and magnitude. On the other hand, we have examples like (58, 80) labeled as 0, where the numbers are further apart. Given these observations, the closeness of the numbers in our input (56, 59) seems to be the most significant factor, leading us to predict a label of 1 for this pair.
Label: 1
Input: 49 69
Steps: To predict the label for the input "27 39," I look at the given pairs and their labels, focusing on the relationship between the two numbers in each pair. When the numbers in a pair are not too far apart, especially when the difference is moderate and both numbers are mid-range or above, the label tends to be '1'. For example, pairs like "66 30" and "67 65" have labels of '1' even though their numbers are not extremely close, but they are within a reasonable range of each other. In the case of "27 39," the difference between the two numbers is 12, and both numbers are relatively close, following the trend of other '1' labeled pairs. Based on this pattern, I conclude that the label is '1'.
Label: 1
Input: 46 39
Steps: To predict the label for the input "46 39," I first analyze the given pairs and their labels, focusing on how the numbers relate to each other. I observe that when one number is higher and the other is moderately lower, especially if both numbers are somewhat close to each other and not too small, the label tends to be '1'. For instance, the pair "66 30" has a label of '1', even though there's a noticeable difference between the two numbers. In the case of "46 39," the numbers are fairly close, with a difference of 7, and both numbers are moderate, which follows the pattern observed in other pairs labeled '1'. Therefore, I predict that the label for "46 39" is '1'.
Label: 1
Input: 20 44
Steps: Let's analyze the input pair (20, 44) in the context of the patterns observed in the given data. First, we notice that there's a significant difference between these two numbers, with 44 being more than twice as large as 20. Looking at previous examples, we see that pairs with larger differences between the numbers, especially where the second number is notably larger, tend to be labeled as 0. For instance, (25, 52) and (22, 61) both follow this pattern and are labeled 0. Additionally, we should consider the overall magnitude of these numbers. Both 20 and 44 fall on the lower end of the range we've seen in the dataset. This aligns with examples like (29, 54) which was also labeled 0. The combination of a significant difference between the numbers and their relatively low values strongly suggests a pattern consistent with the label 0. Therefore, based on these observations and comparisons to similar examples in the dataset, we can reasonably predict that the label for (20, 44) is likely to be 0.
Label: 0"""
    return prompt

pattern = re.compile(r"Label:\s*([01])")

def parse_label(generated_text):
    match = pattern.search(generated_text)
    if match:
        label = int(match.group(1))
        print(f"Predicted label: {label}")
    else:
        print("Label not found in generated text.")
        label = None
    return label

def generate_dataset(args, meta_train_X, meta_train_y):
    """Generate context and query datasets for training and testing."""
    context_x = []
    context_y = []
    query_x = []
    query_y = []
    reasoning_x = []
    reasoning_y = []

    for task_idx, (task_x, task_y) in enumerate(zip(meta_train_X, meta_train_y)):
        num_per_class = args.num_in_context // 2 + args.num_test_samples // 2 + args.num_in_context_reasoning // 2
        class_0_indices = np.where(task_y == 0)[0][:num_per_class]
        class_1_indices = np.where(task_y == 1)[0][:num_per_class]
        context_0_indices = class_0_indices[: args.num_in_context // 2]
        context_1_indices = class_1_indices[: args.num_in_context // 2]
        test_0_indices = class_0_indices[args.num_in_context // 2 : args.num_in_context // 2 + args.num_test_samples // 2]
        test_1_indices = class_1_indices[args.num_in_context // 2 : args.num_in_context // 2 + args.num_test_samples // 2]
        reasoning_0_indices = class_0_indices[args.num_in_context // 2 + args.num_test_samples // 2 :]
        reasoning_1_indices = class_1_indices[args.num_in_context // 2 + args.num_test_samples // 2 :]
        context_indices = np.concatenate([context_0_indices, context_1_indices])
        test_indices = np.concatenate([test_0_indices, test_1_indices])
        reasoning_indices = np.concatenate([reasoning_0_indices, reasoning_1_indices])
        np.random.shuffle(context_indices)
        np.random.shuffle(reasoning_indices)

        context_x.append(task_x[context_indices])
        context_y.append(task_y[context_indices])
        query_x.append(task_x[test_indices])
        query_y.append(task_y[test_indices])
        reasoning_x.append(task_x[reasoning_indices])
        reasoning_y.append(task_y[reasoning_indices])

        # Ensure no overlap between context and query sets
        assert len(set(context_indices) & set(test_indices) & set(reasoning_indices)) == 0

    print("Generated context and query datasets.")
    return np.array(context_x), np.array(context_y), np.array(query_x), np.array(query_y), np.array(reasoning_x), np.array(reasoning_y)
