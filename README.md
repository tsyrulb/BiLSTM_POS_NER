# BiLSTM POS and NER Tagging (NLP Course Homework)

This repository was created as part of a deep learning in NLP course. It implements several BiLSTM-based models for Part-of-Speech (POS) and Named Entity Recognition (NER) tagging, including versions with character-level embeddings and pre-trained word embeddings.

## Quick Start

1. **Clone & Install**  
   ```bash
   git clone https://github.com/tsyrulb/BiLSTM_POS_NER.git
   cd BiLSTM_POS_NER
   pip install -r requirements.txt
   ```

2. **Train**  
   ```bash
   python bilstmTrain.py <model> <train_file> <model_file> [options]
   ```
   - Example: `python bilstmTrain.py c pos/train model_pos_c.pth --epochs 10`

3. **Predict**  
   ```bash
   python bilstmPredict.py <model> <model_file> <input_file> [options]
   ```
   - Example: `python bilstmPredict.py c model_pos_c.pth pos/test --outputFile predictions.pos`

## Models

- **BiLSTM**  
- **BiLSTMWithChar**  
- **BiLSTMWithPretrained**  
- **BiLSTMWithCharAndWord**  

Each variant handles embeddings (word-level, character-level, or both) differently. Check the scripts for more details on how to configure each model.
