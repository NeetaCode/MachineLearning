📘 ANN_Sentiment_Analysis.ipynb:
************************************************************************************************************************************
This notebook explores a series of deep learning models for sentiment analysis, focusing on progressively refining architectures and evaluating their performance. The journey spans from simple RNNs to more advanced recurrent structures and tokenization techniques using BERT.

🔧 1. Initial RNN Model with ReLU Activation
•	Architecture: Embedding → SimpleRNN(32, activation='relu') → Dense(softmax)
•	Training: 5 epochs, categorical cross-entropy, Adam optimizer
•	Evaluation:
  o	Tracked training time
  o	Plotted training and validation accuracy/loss

🔁 2. Deep RNN Architecture
•	Stacked RNN Layers:
  o	SimpleRNN(64, return_sequences=True) → SimpleRNN(32, return_sequences=True) → SimpleRNN(16)
  o	Added Dropout(0.2)
•	Improved Learning Capacity for sequential dependencies
•	Evaluated with similar metrics and time tracking

🧠 3. Replacing RNN with LSTM
•	Layer Stack:
  o	LSTM(64, return_sequences=True) → LSTM(32, return_sequences=True) → LSTM(16)
  o	Used Dropout(0.5)
•	Rationale: LSTM improves long-term memory retention over RNN
•	Results: Better validation accuracy, more stable learning curves

⚡ 4. GRU-Based Architecture
•	Replaced LSTM with GRU:
  o	GRU(64) → GRU(32) → GRU(16)
  o	Added Dropout(0.2)
•	Benefits: Computationally efficient with competitive performance

🔄 5. Bidirectional GRUs
•	Introduced Bidirectional wrappers:
  o	Bidirectional(GRU(64)) → Bidirectional(GRU(32)) → Bidirectional(GRU(16))
•	Goal: Capture context in both forward and backward directions
•	Observation: Enhanced model understanding of sentiment semantics

🧪 6. BERT Tokenization and Input Pipeline
•	Used Hugging Face imdb dataset
•	Tokenized with BertTokenizer (bert-base-uncased)
•	Inputs truncated/padded to max length = 128
•	Encoded labels using LabelEncoder
•	Train/test split using train_test_split

⚙️ 7. Custom Lightweight LSTM with Mixed Precision
•	Architecture:
  o	Embedding → Bidirectional(LSTM(32)) → Bidirectional(LSTM(16)) → Dropout(0.5) → Dense(8, relu) → Dense(softmax)

•	Mixed Precision Training:
  o 	Enabled with tf.keras.mixed_precision
  o	Improved training speed and memory efficiency
•	Training: 3 epochs, binary_crossentropy loss, 20% validation split
•	Visualization: Accuracy and loss plots
•	Evaluation: Measured final test accuracy and loss

📊 Performance Tracking
Across all models:
	📈 Plotted training & validation metrics
	⏱️ Measured total training time
	📉 Evaluated final model loss and accuracy on the test set
