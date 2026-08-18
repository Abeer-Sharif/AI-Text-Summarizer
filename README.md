# 🤖 AI Text Summarizer

### Transformer-Based Text Summarization using T5

An AI-powered text summarization application that uses a **T5 Transformer model** to generate concise summaries from long-form text and dialogues.

The project demonstrates an end-to-end NLP application using **Python, PyTorch, Hugging Face Transformers, FastAPI, and a web-based frontend**.

<img width="1283" height="862" alt="image" src="https://github.com/user-attachments/assets/59bf708d-2485-43b6-be6d-65f6f824f4a7" />


---

## ✨ Overview

Reading and processing large amounts of text can be time-consuming. This project uses a Transformer-based language model to identify important information and generate a shorter summary while preserving the core meaning of the original text.

### 🎯 Workflow

```text
User Input
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
T5 Transformer
    ↓
Text Generation
    ↓
Generated Summary
```

---

## 🚀 Features

* 🧠 **T5 Transformer-based summarization**
* 📝 Summarization of text and dialogues
* 🔤 Hugging Face tokenization
* ⚡ PyTorch-based inference
* 🌐 FastAPI backend
* 💻 Web-based user interface
* ⏳ Loading state during summary generation
* ✂️ Input truncation for long text
* 🎯 Configurable summary length
* 📱 Responsive interface

---

## 🛠️ Tech Stack

| Category        | Technologies              |
| --------------- | ------------------------- |
| Language        | Python                    |
| Deep Learning   | PyTorch                   |
| NLP             | Hugging Face Transformers |
| Model           | T5 Transformer            |
| Backend         | FastAPI                   |
| Server          | Uvicorn                   |
| Frontend        | HTML, CSS, JavaScript     |
| Development     | VS Code                   |
| Version Control | Git & GitHub              |

---

## 🧠 Model Architecture

The application uses the **T5 (Text-to-Text Transfer Transformer)** architecture.

T5 approaches different NLP tasks as text-to-text problems. For this project, the model is used for abstractive text summarization.

```text
Input Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
T5 Encoder
    ↓
T5 Decoder
    ↓
Generated Token IDs
    ↓
Decoded Summary
```

---

## 🔍 How It Works

### 1. Input

The user enters a text or dialogue that needs to be summarized.

Example:

```text
Reporter: How is artificial intelligence changing healthcare?

Expert: AI can process large amounts of medical data much faster
than humans. It can help doctors identify patterns in medical images,
predict potential health risks, and support clinical decision-making.
```

### 2. Preprocessing

The input text is cleaned and prepared before being passed to the tokenizer.

### 3. Tokenization

The Hugging Face tokenizer converts the input text into numerical token IDs.

```python
inputs = tokenizer(
    dialogue,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_tensors="pt"
)
```

The input is limited to a maximum of **512 tokens** and truncated when necessary.

### 4. Text Generation

The tokenized input is passed to the T5 model to generate the summary.

```python
summary_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens=100
)
```

### 5. Decoding

The generated token IDs are decoded back into readable text and returned to the user.

---

## 💻 Project Structure

```text
AI-Text-Summarizer/
│
├── app.py
├── index.html
├── style.css
├── script.js
├── requirements.txt
├── .gitignore
└── README.md
```

> Model weights are excluded from the Git repository because Transformer model files can be very large.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Abeer-Sharif/AI-Text-Summarizer.git
cd AI-Text-Summarizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
uvicorn app:app --reload
```

Open the local URL provided by Uvicorn in your browser.

---

## 📦 Requirements

The main dependencies used in this project are:

```text
fastapi
uvicorn
torch
transformers
sentencepiece
```

The complete dependency list is available in `requirements.txt`.

---

## 🧪 Example

### Input

```text
Reporter: What role can AI play in modern healthcare?

Expert: Artificial intelligence can process large amounts of
medical data quickly. It can assist doctors by identifying
patterns in medical images, predicting potential health risks,
and supporting clinical decision-making.
```

### Generated Summary

```text
AI can analyze medical data, identify patterns in medical images,
predict health risks, and support doctors in clinical decisions.
```

---

## 📊 Key Concepts

This project provides hands-on experience with:

* Transformer architecture
* T5 models
* Sequence-to-sequence learning
* Natural Language Processing
* Text tokenization
* Attention mechanisms
* PyTorch
* Hugging Face Transformers
* Text generation
* Model inference
* FastAPI
* Frontend-backend integration

---

## 🎯 Why T5?

T5 uses the Transformer architecture and represents NLP tasks as **text-to-text** problems.

It can be applied to tasks such as:

* Text summarization
* Translation
* Question answering
* Text generation
* Text classification

In this project, T5 is used for **abstractive summarization**, meaning the model generates a new summary based on the input rather than simply selecting existing sentences.

---

## ⚠️ Limitations

* Long inputs may need to be truncated due to model input limits.
* CPU inference can be slower than GPU inference.
* Generated summaries may occasionally omit important information.
* AI-generated summaries may contain inaccuracies.
* Summary quality depends on the selected model.

---

## 🔮 Future Improvements

* Deploy the application publicly
* Add PDF and document upload
* Support multiple languages
* Add adjustable summary length
* Improve long-document summarization
* Host the trained model on Hugging Face Hub

---

## 🌐 Deployment

The application can be deployed using platforms such as:

* Hugging Face Spaces
* Render
* Railway
* AWS

The trained model should be hosted separately rather than committed directly to GitHub.

---

## ⭐ Project Highlights

```text
✔ Transformer-based NLP application
✔ T5 text summarization
✔ PyTorch inference
✔ Hugging Face Transformers
✔ FastAPI backend
✔ Custom web interface
✔ End-to-end AI application
```

---

## 📌 Project Status

🟢 **Active Development**

The core summarization pipeline and web interface are implemented. Future development will focus on improving long-text handling, deployment, and model accessibility.

---
## 👩‍💻 Author

### Abeer Sharif

**B.E. Electronics and Computer Science**


⭐ If you found this project useful, consider giving the repository a star.
