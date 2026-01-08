# EmotiNet Experimentation Log

## Data Preprocessing
- Max vocabulary: 10,000 words
- Max sequence length: 50
- Padding strategy: post
- Truncation strategy: post

## Class Balancing
- Applied compute_class_weight('balanced') to counteract emotion sample imbalance.

## Model 1: Simple RNN
- Embedding(10000, 128, input_length=50)
- SimpleRNN(128, return_sequences=True) + Dropout(0.5)
- SimpleRNN(64) + Dropout(0.5)
- Dense(6, softmax)
- Simple RNN Test Loss: 0.689, Test Accuracy: 76.1%
- Observation: Slower gradient propagation, struggles with long dependencies.
