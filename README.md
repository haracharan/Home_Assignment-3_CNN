# Home_Assignment-3_CNN
# Deep Learning Projects using TensorFlow & Keras

This repository contains four deep learning projects built with TensorFlow and Keras, covering key concepts like autoencoders, recurrent neural networks (RNNs), and natural language processing (NLP). Each project demonstrates practical implementation and visualization techniques.

---

## 📁 Projects Included

### 1. Basic Autoencoder for Image Reconstruction
A fully connected autoencoder trained on the MNIST dataset to learn compressed representations and reconstruct images.

- **Model**: Dense → Dense (32 latent dim) → Dense
- **Dataset**: MNIST
- **Loss**: Binary cross-entropy
- **Features**:
  - Visualization of reconstructed images
  - Analysis of different latent dimensions (16, 32, 64)

🔗 File: `basic_autoencoder.py`

---

### 2. Denoising Autoencoder
Enhances the autoencoder to reconstruct clean images from noisy inputs.

- **Noise**: Gaussian (mean=0, std=0.5)
- **Model**: Same as basic autoencoder
- **Features**:
  - Visualization of noisy vs. denoised images
  - Comparison with basic autoencoder
  - Real-world use case: Medical imaging

🔗 File: `denoising_autoencoder.py`

---

### 3. LSTM-based Text Generation (Character-level)
Uses an LSTM RNN to learn patterns in text (The Little Prince) and generate new text sequences.

- **Model**: Embedding → LSTM → Dense
- **Dataset**: The Little Prince (Gutenberg)
- **Features**:
  - Character-level prediction
  - Temperature scaling to control randomness
  - Sample outputs at different temperatures

🔗 File: `text_generation_rnn.py`

---

### 4. Sentiment Classification using LSTM
Classifies IMDB movie reviews as positive or negative using a sequence-based LSTM model.

- **Model**: Embedding → LSTM → Dense (sigmoid)
- **Dataset**: IMDB (pre-tokenized)
- **Evaluation**:
  - Confusion matrix
  - Accuracy, Precision, Recall, F1-score
- **Concepts**:
  - Precision-recall tradeoff interpretation

🔗 File: `sentiment_classification_lstm.py`

---

## 🔧 Requirements

Install all required packages using:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
