import heapq
from typing import List, Tuple
from ranking_predictor import RankingPredictor
from request import Request

class RankingScheduler:
    def __init__(self, predictor: RankingPredictor, starvation_threshold: int = 5, priority_quantum: int = 3):
        self.predictor = predictor
        self.starvation_threshold = starvation_threshold
        self.priority_quantum = priority_quantum
        self.request_queue: List[Tuple[float, float, Request]] = []

    def add_requests(self, new_requests: List[Request]):
        prompts = [req.prompt for req in new_requests]
        scores = self.predictor.predict_scores(prompts)
        for req, score in zip(new_requests, scores):
            req.score = score
            heapq.heappush(self.request_queue, (-score, req.arrival_time, req))

    def get_next_batch(self, batch_size: int, max_tokens: int) -> List[Request]:
        batch = []
        token_count = 0
        temp_queue = []

        while self.request_queue and len(batch) < batch_size and token_count < max_tokens:
            _, _, req = heapq.heappop(self.request_queue)
            
            if req.priority:
                req.quantum -= 1
                batch.append(req)
                token_count += len(req.prompt.split())  # Rough estimation of token count
            else:
                temp_queue.append((-req.score, req.arrival_time, req))

        # If batch is not full, add non-priority requests
        while temp_queue and len(batch) < batch_size and token_count < max_tokens:
            _, _, req = heapq.heappop(temp_queue)
            batch.append(req)
            token_count += len(req.prompt.split())

        # Put back unused requests
        for item in temp_queue:
            heapq.heappush(self.request_queue, item)

        # Update starvation counts and priorities
        for _, _, req in self.request_queue:
            if req not in batch:
                req.starvation_count += 1
                if req.starvation_count >= self.starvation_threshold:
                    req.priority = True
                    req.quantum = self.priority_quantum
                    req.starvation_count = 0
            elif req.priority and req.quantum <= 0:
                req.priority = False

        return batch

    def remove_finished_requests(self, finished_requests: List[Request]):
        self.request_queue = [item for item in self.request_queue if item[2] not in finished_requests]
        heapq.heapify(self.request_queue)
