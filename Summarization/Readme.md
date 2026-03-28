# Text Summarization Fine-Tuning with Hugging Face Transformers

This repository contains example notebook `ml_training.ipynb` for fine‑tuning state‑of‑the‑art transformer models on an abstractive summarization task using the [Hugging Face Transformers](https://github.com/huggingface/transformers) library. The repository demonstrates how to fine‑tune several models on a summarization dataset and also provides a web interface for interactive inference.

Due to GPU shortages, I was only able to fine‑tune the **T5‑Small** model. The Pegasus‑XSum and BART‑Base experiments were not completed because of limited GPU resources.  
![GPU Shortage](https://gitlab.informatik.uni-bremen.de/alimirza/aml_project/-/raw/main/Summarization/GPU_Shortage.png)  
*Screenshot: GPU shortage – only T5‑Small was fine‑tuned due to hardware constraints.*

---

## Table of Contents

- [Overview](#overview)
- [Abstractive Vs Extractive](#abstractive)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [T5‑Small Fine‑Tuning](#t5-small-fine-tuning)
  - [Pegasus‑XSum Fine‑Tuning](#pegasus-xsum-fine-tuning)
  - [BART‑Base Fine‑Tuning](#bart-base-fine-tuning)
- [Web Interface](#web-interface)
- [Architectural Distinctions and Design Philosophies](#architectural-distinctions-and-design-philosophies)
- [Preprocessing Innovations](#preprocessing-innovations)
- [Code Explanation](#code-explanation)
  - [Compute Metrics Function](#compute-metrics-function)
  - [Preprocessing Function](#preprocessing-function)
  - [Training Arguments and Trainer Setup](#training-arguments-and-trainer-setup)
- [Resuming from Checkpoints](#resuming-from-checkpoints)
- [Evaluation](#evaluation)
- [License](#license)

---

## Overview

This project demonstrates how to:
- Preprocess text data (articles and highlights) by adding a task-specific prefix (e.g. `"summarize: "` for T5).
- Tokenize the inputs and targets with appropriate padding, truncation, and maximum lengths.
- Define a custom `compute_metrics` function that calculates ROUGE scores and generation length.
- Fine‑tune three different sequence‑to‑sequence models using Hugging Face’s `Trainer` API.

Each model is fine‑tuned on a summarization task (e.g. CNN/DailyMail) with consistent evaluation metrics, making it easy to compare results.

---

## Abstractive Vs Extractive

**Extractive Summarization** selects and combines the most important sentences or phrases directly from the source text without altering the original wording. For example, consider this article snippet:

> "Scientists at NOAA reported that global temperatures have risen by 1.2°C over the past century due to increased greenhouse gas emissions. This warming has resulted in melting polar ice, rising sea levels, and more frequent extreme weather events. The study calls for urgent action to reduce emissions and mitigate future warming."

**An extractive summary** might simply pick key sentences like:

> "Global temperatures have risen by 1.2°C over the past century due to increased greenhouse gas emissions, leading to melting polar ice, rising sea levels, and more frequent extreme weather events."

This approach is particularly useful in contexts where preserving the exact language is critical—such as legal documents, medical reports, or when quotes need to be maintained.

**Abstractive Summarization** goes a step further by generating a new text that paraphrases the source material while conveying its essential meaning. Using the same article snippet, **an abstractive summar**y could be:

> "Scientists warn that human-driven emissions have significantly warmed the planet, causing ice melt, sea-level rise, and severe weather, and stress the need for immediate action to curb climate change."

This method is beneficial when a concise, coherent, and human-like summary is desired—such as in news briefs, research paper abstracts, or social media summaries—because it can integrate information from various parts of the text and rephrase it in a more natural or succinct manner

---

## Prerequisites

- Python 3.8+
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [Evaluate](https://github.com/huggingface/evaluate)
- [NumPy](https://numpy.org/)
- [Streamlit](https://streamlit.io/) (for the web interface)

Make sure you have access to a GPU (or adjust settings for CPU training) and configure your device appropriately (e.g. using `torch.device("cuda")`).

---

## Usage

Each section below shows a different training script for a specific model. These examples are provided in Jupyter Notebook format (e.g. `ml_training.ipynb`) but can be adapted into standalone Python scripts.

### T5‑Small Fine‑Tuning

- **Checkpoint:** `google-t5/t5-small`
- **Process:**
  - Loads the T5‑small tokenizer and model.
  - Adds the `"summarize: "` prefix to every article.
  - Tokenizes articles (max_length=512) and highlights (max_length=128).
  - Sets up training arguments (learning rate, batch size, epochs, etc.).
  - Creates a `Seq2SeqTrainer` and calls `trainer.train()`.

Below is an updated section you can add to your README (or "canvas") that describes the fine‑tuned T5 model performance:

---

### Pegasus‑XSum Fine‑Tuning

- **Checkpoint:** `google/pegasus-xsum`
- **Process:**
  - Loads the Pegasus‑XSum tokenizer and model.
  - Applies the same preprocessing function for tokenizing inputs and targets.
  - Uses a `DataCollatorForSeq2Seq` to prepare batches.
  - Defines training arguments (with a smaller batch size due to model size) and runs training via `trainer.train(resume_from_checkpoint=True)`.

### BART‑Base Fine‑Tuning

- **Checkpoint:** `facebook/bart-base`
- **Process:**
  - Loads the BART‑base tokenizer and model.
  - Moves the model to GPU (using `model.to(device)`).
  - Sets up training arguments optimized for CPU (with reduced batch size, gradient accumulation steps, etc.).
  - Instantiates a `Trainer` and starts training with `trainer.train(resume_from_checkpoint=True)`.

---

## Web Interface

This repository also contains a web interface built with [Streamlit](https://streamlit.io/), allowing users to interact with different summarization models. To run the web interface:

1. Ensure you have a valid Hugging Face API key.
2. Run the following command from the repository's root directory:

   ```bash
   streamlit run app.py
   ```

The `app.py` script provides:
- A sidebar for entering your Hugging Face API key.
- A model selection dropdown with several available models (e.g., fine‑tuned T5‑small, Pegasus‑CNN/DailyMail variants, etc.).
- Example texts that you can use to test the summarization.
- A text area for entering your own text.
- A "Summarize" button that generates and displays the summary.

Below is an excerpt from `app.py`:

```python
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Sidebar for API key input
st.sidebar.title("Settings")
hf_api_key = st.sidebar.text_input("Enter your Hugging Face API Key", type="password")
max_length = st.sidebar.slider("Max Summary Length", min_value=50, max_value=300, value=150, step=10)

# Model selection
st.title("Abstractive Text Summarization")
st.write("Select a model and enter text below to generate a summary.")

model_choice = st.selectbox("Choose a model:", [
    "k200353/t5-small-finetuned-cnn-dailymail",
    "facebook/bart-large-cnn",
    "t5-small",
    "t5-base",
    "google/pegasus-cnn_dailymail",
    "google/pegasus-xsum",
    "k200353/pegasus-finetuned-cnn_dailymail"
])

# Load the selected model with API key
if hf_api_key:
    tokenizer = AutoTokenizer.from_pretrained(model_choice, token=hf_api_key)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_choice, token=hf_api_key)
else:
    st.sidebar.warning("Please enter your Hugging Face API key to load the model.")
    model = None

# Example texts
examples = { ... }  # (Example texts omitted for brevity)

input_text = st.text_area("Enter your text:", height=200)

if st.button("Summarize"):
    if model and input_text:
        with st.spinner("Generating summary..."):
            tokens = tokenizer(input_text, truncation=True, padding="longest", return_tensors="pt")
            summary = model.generate(**tokens)
            decoded_output = tokenizer.decode(summary[0], skip_special_tokens=True)
            st.subheader("Generated Summary")
            st.write(decoded_output)
    elif not model:
        st.warning("Model not loaded. Please enter a valid Hugging Face API key.")
    else:
        st.warning("Please enter text before summarizing.")
```

---

## Architectural Distinctions and Design Philosophies

### T5‑Base: The Text-to-Text Generalist

T5 reimagines NLP tasks through a unified text-to-text framework where inputs and outputs are always strings of text. Unlike BERT-style models restricted to classification or span prediction, T5 reformulates summarization as a sequence-to-sequence task with a **“summarize:” prefix**. Its architecture follows a standard encoder-decoder transformer but introduces two critical innovations:

- **Span Corruption Pretraining:**  
  During pretraining, T5 masks consecutive token spans replaced by a single sentinel token, forcing the model to reconstruct entire phrases rather than individual words [^1].

- **Task-Specific Prefixes:**  
  All downstream tasks are prepended with task identifiers (e.g., “translate English to German:”), enabling multi-task learning without architectural modifications [^1].

### Pegasus‑XSum: Gap-Sentence Generation Specialist

Pegasus adopts an encoder-decoder architecture pretrained using **Gap-Sentence Generation (GSG)**, a novel objective where approximately 30% of salient sentences are masked and regenerated. This approach mimics summarization by requiring the model to identify and reconstruct key content.

- **Importance Sampling:**  
  Sentences are selected for masking based on ROUGE-F1 scores against the original document, prioritizing informationally dense content [^2].

- **Mixture of Datasets:**  
  Pretraining combines C4 (800GB of web text) and HugeNews (1.5B news articles), enhancing domain adaptability [^2][^4].

The model uses GeLU activations and cross-entropy loss. Its encoder contains 12 layers with 768-dimensional hidden states, matching BERT-Large’s configuration [^2].

### BART‑Base: Bidirectional Denoising for Reconstruction

BART combines BERT’s bidirectional encoder with GPT-style autoregressive decoding. Its pretraining involves corrupting documents through several noise functions:

- **Text Infilling:**  
  Random text spans are replaced with a single mask token.
- **Sentence Permutation:**  
  The original sentence order is shuffled.
- **Token Deletion:**  
  Arbitrary tokens are removed.

By reconstructing the original text, BART learns robust representations for generation tasks.

- **Key Architectural Trait:**  
  BART uses GeLU activations for smoother gradients [^3].

[^1]: [Abstractive Summarization using Google’s T5](https://turbolab.in/abstractive-summarization-using-googles-t5/)  
[^2]: [Google Pegasus-XSum Model](https://dataloop.ai/library/model/google_pegasus-xsum/)  
[^3]: [Transformers BART Model Explained](https://www.projectpro.io/article/transformers-bart-model-explained/553)  
[^4]: [Coling 2020 Pegasus Paper](https://aclanthology.org/2020.coling-main.494.pdf)

---

## Preprocessing Innovations

### T5‑Base: Task Prefixing and Span Corruption

- **Task Prefixing:**  
  Inputs are prefixed with task descriptors (e.g., `"summarize:"`) to clearly guide the model's generation.
- **Aggressive Data Cleaning:**  
  The pretraining data (C4 corpus) undergoes deduplication, language filtering, and removal of offensive content [^1].

### Pegasus‑XSum: Stochastic Sampling and HugeNews Curation

- **Salient Sentence Selection:**  
  Uses ROUGE-based importance sampling during pretraining to select the most informative sentences.
- **Hybrid Dataset:**  
  Combines the breadth of the C4 corpus with the domain specificity of HugeNews [^2].

### BART‑Base: Document Corruption Strategies

- **Noise Injection:**  
  Pretraining employs text infilling, sentence shuffling, and token deletion to simulate diverse noise patterns.
- **Task-Specific Tokenization:**  
  Fine‑tuning further leverages task-specific prefixes (e.g., “TL;DR”) to adapt to summarization tasks [^3].

---

## Code Explanation

### Compute Metrics Function

The `compute_metrics` function utilizes the [evaluate](https://github.com/huggingface/evaluate) library to calculate ROUGE scores and the average generated token length:

```python
import evaluate
import numpy as np

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}
```

### Preprocessing Function

The `preprocess_function` adds a `"summarize: "` prefix to each article and tokenizes both the input text and the target highlights:

```python
from transformers import AutoTokenizer

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs
```

### Training Arguments and Trainer Setup

Each model is trained with tailored hyperparameters. For instance, T5‑Small training is configured as follows:

```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="t5-small-finetuned-cnn-dailymail",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,  # or use bf16=True for XPU
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=True)
```

Similar training setups are provided for Pegasus‑XSum and BART‑Base.

---

## Resuming from Checkpoints

The training scripts include the argument `resume_from_checkpoint=True` so that if training is interrupted, you can restart from the latest checkpoint without losing progress.

---

## Evaluation


### Fine Tuned T5 Model

| Epoch | Training Loss | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len   |
|-------|---------------|-----------------|---------|---------|---------|-----------|-----------|
| 1     | 1.155500      | 1.043971        | 0.244600| 0.111300| 0.201200| 0.201200  | 19.968800 |
| 2     | 1.144900      | 1.040560        | 0.244700| 0.111400| 0.201400| 0.201500  | 19.968200 |

> **Note:**  
> The fine‑tuned T5 model performance (with only 2 epochs) was not as good as the main T5‑Base model but outperformed the T5‑Small model. This performance could be further improved with a better choice of hyperparameters and additional training epochs. Due to current GPU hardware constraints (Google T4 GPU and Kaggle's 2× T4 GPUs), only 2 epochs were feasible.

![Evaluation Graph](https://gitlab.informatik.uni-bremen.de/alimirza/aml_project/-/raw/main/Summarization/evaluation.png?ref_type=heads)


![Benchmarks](https://gitlab.informatik.uni-bremen.de/alimirza/aml_project/-/raw/main/Summarization/Benchmarks.png?ref_type=heads)  
You can see that Pegasus has the highest Rogue Score, followed by Bart and then T5 model. If given sufficient gpu resources we would train on pegasus, try out different variants of BART model to compare the results. Train the T5 on T5-base and train all of the models on more epochs to properly see the comparison. Note the reason why Pegasus performs better than other models can be understood in the [Architectural Distinctions and Design Philosophies](#architectural-distinctions-and-design-philosophies) i.e Gap-Sentence Generation (GSG) works better than MLM (Masked Language Model)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive guide to fine‑tuning and interacting with summarization models, explains their architectural distinctions and preprocessing strategies, and documents the interactive web interface. Happy fine‑tuning and summarizing!

Abstractive Summarization using Google’s T5 ↩ ↩2 ↩3

Google Pegasus-XSum Model ↩ ↩2 ↩3 ↩4

Coling 2020 Pegasus Paper ↩

Transformers BART Model Explained ↩ ↩2