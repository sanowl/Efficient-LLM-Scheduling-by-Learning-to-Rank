import time
import random
from typing import List, Tuple
from request import Request
import numpy as np

class RequestGenerator:
    def __init__(self, prompt_templates: List[str], arrival_rate: float):
        self.prompt_templates = prompt_templates
        self.arrival_rate = arrival_rate
        self.last_arrival_time = time.time()

    def generate_requests(self) -> List[Request]:
        current_time = time.time()
        time_diff = current_time - self.last_arrival_time
        num_requests = np.random.poisson(self.arrival_rate * time_diff)
        
        requests = []
        for _ in range(num_requests):
            prompt = random.choice(self.prompt_templates)
            arrival_time = self.last_arrival_time + random.uniform(0, time_diff)
            requests.append(Request(prompt=prompt, arrival_time=arrival_time))
        
        self.last_arrival_time = current_time
        return sorted(requests, key=lambda r: r.arrival_time)

def receive_new_requests(request_generator: RequestGenerator) -> List[Request]:
    return request_generator.generate_requests()

def output_response(request: Request, response: str):
    print(f"Response for prompt '{request.prompt}': {response}")
    print(f"Generation time: {time.time() - request.arrival_time:.2f} seconds")
    print(f"Output length: {len(response.split())}\n")

class SimpleLLM:
    def __init__(self, token_generation_time: float = 0.1):
        self.token_generation_time = token_generation_time

    def generate(self, prompt: str) -> str:
        # Simulate token generation time
        output_length = len(prompt.split()) + random.randint(10, 50)
        time.sleep(self.token_generation_time * output_length)
        
        # Generate a simple response
        return f"Generated response for: {prompt} " + " ".join(["token"] * (output_length - len(prompt.split())))

def initialize_llm(token_generation_time: float = 0.1) -> SimpleLLM:
    return SimpleLLM(token_generation_time)

def calculate_kendall_tau(predicted_ranks: List[int], true_ranks: List[int]) -> float:
    n = len(predicted_ranks)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (predicted_ranks[i] < predicted_ranks[j] and true_ranks[i] < true_ranks[j]) or \
               (predicted_ranks[i] > predicted_ranks[j] and true_ranks[i] > true_ranks[j]):
                concordant += 1
            elif (predicted_ranks[i] < predicted_ranks[j] and true_ranks[i] > true_ranks[j]) or \
                 (predicted_ranks[i] > predicted_ranks[j] and true_ranks[i] < true_ranks[j]):
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)

def evaluate_scheduler_performance(actual_times: List[float], predicted_ranks: List[int]) -> Tuple[float, float]:
    true_ranks = list(range(len(actual_times)))
    true_ranks.sort(key=lambda i: actual_times[i])
    
    kendall_tau = calculate_kendall_tau(predicted_ranks, true_ranks)
    
    fcfs_total_time = sum(actual_times)
    scheduled_total_time = sum(t * (i + 1) for i, t in enumerate(sorted(actual_times)))
    
    improvement_ratio = fcfs_total_time / scheduled_total_time
    
    return kendall_tau, improvement_ratio

def generate_synthetic_data(num_samples: int) -> List[Tuple[str, int]]:
    prompt_templates = [
        "Explain {topic} in simple terms.",
        "What are the main differences between {topic1} and {topic2}?",
        "Write a short story about {character} in {setting}.",
        "List the top 5 {category}.",
        "How does {process} work?",
    ]
    
    topics = ["quantum computing", "machine learning", "climate change", "blockchain", "artificial intelligence"]
    characters = ["a detective", "a time traveler", "a robot", "a superhero", "an alien"]
    settings = ["ancient Rome", "a space station", "underwater city", "post-apocalyptic world", "magical forest"]
    categories = ["inventions", "scientific discoveries", "historical events", "books", "movies"]
    processes = ["photosynthesis", "digestion", "cloud formation", "electricity generation", "DNA replication"]
    
    data = []
    for _ in range(num_samples):
        template = random.choice(prompt_templates)
        if "{topic}" in template:
            prompt = template.format(topic=random.choice(topics))
        elif "{topic1}" in template and "{topic2}" in template:
            t1, t2 = random.sample(topics, 2)
            prompt = template.format(topic1=t1, topic2=t2)
        elif "{character}" in template and "{setting}" in template:
            prompt = template.format(character=random.choice(characters), setting=random.choice(settings))
        elif "{category}" in template:
            prompt = template.format(category=random.choice(categories))
        elif "{process}" in template:
            prompt = template.format(process=random.choice(processes))
        
        # Simulate output length based on prompt complexity
        output_length = len(prompt.split()) * random.randint(2, 5)
        
        data.append((prompt, output_length))
    
    return data

if __name__ == "__main__":
    # Test the utility functions
    llm = initialize_llm(token_generation_time=0.05)
    
    prompt_templates = [
        "Tell me a joke about {topic}",
        "Explain {topic} like I'm five",
        "Write a haiku about {topic}",
    ]
    request_gen = RequestGenerator(prompt_templates, arrival_rate=2.0)
    
    for _ in range(3):  # Simulate 3 batches of requests
        requests = receive_new_requests(request_gen)
        print(f"Received {len(requests)} new requests:")
        for req in requests:
            response = llm.generate(req.prompt)
            output_response(req, response)
        
        time.sleep(1)  # Wait for 1 second before next batch
    
    # Test synthetic data generation
    synthetic_data = generate_synthetic_data(5)
    print("\nSynthetic Data Sample:")
    for prompt, length in synthetic_data:
        print(f"Prompt: {prompt}")
        print(f"Simulated output length: {length}\n")
    
    # Test scheduler performance evaluation
    actual_times = [3.2, 1.5, 4.8, 2.1, 3.7]
    predicted_ranks = [3, 1, 4, 2, 5]
    kendall_tau, improvement_ratio = evaluate_scheduler_performance(actual_times, predicted_ranks)
    print(f"Kendall's Tau: {kendall_tau:.4f}")
    print(f"Improvement Ratio: {improvement_ratio:.4f}")
