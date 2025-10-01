import time
from typing import List, Dict, Callable
import numpy as np

class LLMResponseTrace:
    def __init__(self, prompt: str, response: str, timestamp: float, metrics: Dict[str, float]):
        self.prompt = prompt
        self.response = response
        self.timestamp = timestamp
        self.metrics = metrics

class Evaluator:
    def __init__(self, metric_functions: Dict[str, Callable[[str, str], float]]):
        self.metric_functions = metric_functions
        self.traces: List[LLMResponseTrace] = []

    def evaluate(self, prompt: str, ground_truth: str, response: str) -> LLMResponseTrace:
        timestamp = time.time()
        metrics = {name: func(ground_truth, response) for name, func in self.metric_functions.items()}
        trace = LLMResponseTrace(prompt, response, timestamp, metrics)
        self.traces.append(trace)
        return trace

    def summary(self) -> Dict[str, float]:
        if not self.traces:
            return {}
        metric_names = self.metric_functions.keys()
        result = {}
        for name in metric_names:
            values = [trace.metrics[name] for trace in self.traces]
            result[f"{name}_mean"] = np.mean(values)
            result[f"{name}_std"] = np.std(values)
        return result

# Example metric functions
def accuracy_metric(gt: str, resp: str) -> float:
    return float(gt.strip().lower() == resp.strip().lower())

def length_similarity(gt: str, resp: str) -> float:
    return 1.0 - abs(len(gt) - len(resp)) / max(len(gt), 1)

def jaccard_metric(gt: str, resp: str) -> float:
    gt_set, resp_set = set(gt.lower().split()), set(resp.lower().split())
    intersection = len(gt_set & resp_set)
    union = len(gt_set | resp_set)
    return intersection / union if union else 0.0

# Usage example
if __name__ == "__main__":
    evaluator = Evaluator(metric_functions={
        "accuracy": accuracy_metric,
        "length_similarity": length_similarity,
        "jaccard": jaccard_metric,
    })

    samples = [
        ("What is the capital of France?", "Paris", "Paris"),
        ("What is 2+2?", "4", "Four"),
        ("Describe photosynthesis.", "Conversion of light to energy", "Plants convert sunlight into energy."),
    ]

    for prompt, gt, resp in samples:
        trace = evaluator.evaluate(prompt, gt, resp)
        print(f"Prompt: {prompt}\nResponse: {resp}\nMetrics: {trace.metrics}\nTime: {trace.timestamp}\n")

    print("Summary stats:", evaluator.summary())
