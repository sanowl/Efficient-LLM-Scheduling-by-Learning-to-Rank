from ranking_predictor import RankingPredictor
from ranking_scheduler import RankingScheduler
from utils import receive_new_requests, output_response, initialize_llm
import time

def main_serving_loop(llm, scheduler: RankingScheduler, batch_size: int, max_tokens: int):
    while True:
        # Receive new requests
        new_requests = receive_new_requests()
        scheduler.add_requests(new_requests)

        # Get next batch of requests to process
        batch = scheduler.get_next_batch(batch_size, max_tokens)

        # Process batch with LLM
        finished_requests = []
        for request in batch:
            start_time = time.time()
            response = llm.generate(request.prompt)
            end_time = time.time()
            
            request.output_length = len(response.split())  # Rough estimation of output length
            output_response(request, response)
            
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            finished_requests.append(request)

        # Remove finished requests
        scheduler.remove_finished_requests(finished_requests)

if __name__ == "__main__":
    predictor = RankingPredictor()
    scheduler = RankingScheduler(predictor)
    llm = initialize_llm()
    main_serving_loop(llm, scheduler, batch_size=32, max_tokens=4096)
