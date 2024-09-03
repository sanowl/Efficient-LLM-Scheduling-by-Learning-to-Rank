# LLM Ranking Scheduler

## Overview

This project implements an efficient scheduling system for Large Language Model (LLM) inference, based on the research paper "Efficient LLM Scheduling by Learning to Rank" by Fu et al. The system uses a learning-to-rank approach to predict the relative output lengths of LLM requests, enabling more efficient scheduling and reducing latency in high-load scenarios.

The paper introduces a novel scheduler for LLM inference and serving that can approximate the shortest-job-first (SJF) schedule better than existing approaches. By learning to rank the output lengths of requests, the system can significantly improve performance in important applications, such as chatbot serving and synthetic data generation.

## Features

- Ranking-based scheduler for LLM requests
- Learning-to-rank predictor using a small OPT model
- ListMLE loss for training the ranking model
- Starvation prevention mechanism
- Priority-based scheduling
- Batch processing of requests
- Realistic request generation and LLM simulation
- Performance evaluation using Kendall's Tau and improvement ratio
- Synthetic data generation for training and testing

## Project Structure

The project consists of the following main components:

1. `ranking_predictor.py`: Implements the ranking model using a small OPT model.
2. `request.py`: Defines the Request class to represent individual LLM requests.
3. `ranking_scheduler.py`: Implements the core scheduling logic.
4. `utils.py`: Contains utility functions for request generation, LLM simulation, and performance evaluation.
5. `main.py`: The main serving loop that ties all components together.
6. `train.py`: Script for training the ranking predictor.

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- NumPy

## Installation

Install the required packages:
```
pip install torch transformers numpy
```

## Usage

### Training the Ranking Predictor

1. Prepare your training data or use the synthetic data generation function in `utils.py`.
2. Run the training script:
   ```
   python train.py
   ```
3. The trained model will be saved as `trained_ranking_predictor.pth`.

### Running the LLM Serving System

1. Adjust the parameters in `main.py` as needed (e.g., batch size, max tokens).
2. Run the main serving loop:
   ```
   python main.py
   ```

## Customization

- Modify the `RequestGenerator` in `utils.py` to match your expected request patterns.
- Adjust the `SimpleLLM` class in `utils.py` to more closely simulate your actual LLM's behavior.
- Fine-tune the scheduling parameters in `ranking_scheduler.py` for your specific use case.

## Performance Evaluation

The system includes built-in performance evaluation metrics:

- Kendall's Tau: Measures the correlation between predicted and actual request rankings.
- Improvement Ratio: Compares the performance of the ranking scheduler against a simple FCFS scheduler.

You can use these metrics to assess and fine-tune the scheduler's performance for your specific workload.

## Contributing

Contributions to improve the scheduler or extend its functionality are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## References

This project is based on the following research paper:

Fu, Y., Zhu, S., Su, R., Qiao, A., Stoica, I., & Zhang, H. (2024). Efficient LLM Scheduling by Learning to Rank. arXiv preprint arXiv:2408.15792v1.

## Acknowledgments

- The authors of the "Efficient LLM Scheduling by Learning to Rank" paper for their novel approach to LLM scheduling.
- The open-source community for providing essential libraries like PyTorch and Transformers.
