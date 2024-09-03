LLM Ranking Scheduler
Overview

This project implements an efficient scheduling system for Large Language Model (LLM) inference, based on the paper "Efficient LLM Scheduling by Learning to Rank". The system uses a learning-to-rank approach to predict the relative output lengths of LLM requests, enabling more efficient scheduling and reducing latency in high-load scenarios.
Features

Ranking-based scheduler for LLM requests
Learning-to-rank predictor using a small OPT model
ListMLE loss for training the ranking model
Starvation prevention mechanism
Priority-based scheduling
Batch processing of requests
Realistic request generation and LLM simulation
Performance evaluation using Kendall's Tau and improvement ratio
Synthetic data generation for training and testing

Project Structure
The project consists of the following main components:

ranking_predictor.py: Implements the ranking model using a small OPT model.
request.py: Defines the Request class to represent individual LLM requests.
ranking_scheduler.py: Implements the core scheduling logic.
utils.py: Contains utility functions for request generation, LLM simulation, and performance evaluation.
main.py: The main serving loop that ties all components together.
train.py: Script for training the ranking predictor.

Requirements

Python 3.7+
PyTorch
Transformers library
NumPy
