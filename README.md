# Answer Geography Questions using Machine Learning

A semantic parsing system that translates natural language geography questions into executable logical forms using the [GeoQuery](https://www.cs.utexas.edu/users/ml/nldata/geoquery.html) benchmark dataset. The system can answer questions like *"What states border Texas?"* or *"What is the longest river in the USA?"* by converting them into structured database queries evaluated against a geography knowledge base.

## What It Does

The system performs **natural language to logical form translation** — a core task in semantic parsing. Given a plain English question about U.S. geography, it outputs a structured logical expression that can be executed against a knowledge base to retrieve the answer.

**Example:**
```
Input:  "what states border texas?"
Output: _answer(NV, (_state(V0), _next_to(V0, NV), _const(V0, _stateid(texas))))
```

The predicted logical form is then executed against a Java-backed geography knowledge base to compute the actual answer and measure denotation accuracy.

## Architecture

Two models are implemented:

### 1. Nearest Neighbor Baseline (`NearestNeighborSemanticParser`)
A simple retrieval-based model. For each test question, it finds the most similar training question using **Jaccard similarity** over word overlap, then returns the logical form of that nearest neighbor as the prediction. Serves as a non-learned baseline.

### 2. Sequence-to-Sequence Neural Model (`Seq2SeqSemanticParser`)
A learned encoder-decoder architecture built with PyTorch:

- **Encoder:** A bidirectional LSTM that reads the tokenized input question word by word and produces a fixed-size context vector (final hidden + cell states).
- **Decoder:** A unidirectional LSTM that autoregressively generates output tokens one at a time, initialized from the encoder's final states.
- **Embeddings:** Separate learned embedding layers for input and output vocabularies, with dropout regularization.
- **Training:** Teacher forcing — the gold output token is fed as input at each decoding step during training.
- **Inference:** Greedy decoding — the highest-probability token at each step is selected and fed as the next input, until an `<EOS>` token is produced or a maximum length is reached.
- **Optimizer:** Adam with a learning rate of 0.001.
- **Loss:** Negative log-likelihood (NLLLoss) averaged over output tokens.

## Evaluation

Predictions are evaluated along three dimensions:

| Metric | Description |
|---|---|
| **Exact Match** | Predicted logical form exactly matches the gold logical form (string equality) |
| **Token-level Accuracy** | Position-by-position token overlap between prediction and gold |
| **Denotation Match** | Predicted query executes to the same answer as the gold query against the knowledge base |

Denotation matching is the most meaningful metric — it uses a Java backend (`evaluator/geoquery`) to run both the gold and predicted logical forms against the GeoQuery knowledge base and compares their outputs.

## Dataset

The system uses the **GeoQuery** dataset, a classic semantic parsing benchmark containing ~880 English questions about U.S. geography paired with logical form queries. The data is split into:

- `data/geo_train.tsv` — training set
- `data/geo_dev.tsv` — development/validation set
- `data/geo_test.tsv` — blind test set

Each file is tab-separated with columns: `[natural language question] \t [logical form]`.

Logical forms use De Bruijn variable indexing (standardized by `geoquery_preprocess_lf`) to normalize variable naming and simplify parsing.

## Technologies Used

| Technology | Role |
|---|---|
| **Python 3** | Primary language |
| **PyTorch** (`torch`, `torch.nn`) | Neural network framework — LSTM layers, embedding layers, loss functions, autograd |
| **NumPy** | Array manipulation, padding tensors, computing input lengths |
| **Java** (external binary) | Evaluator backend — executes logical forms against the GeoQuery knowledge base |
| **GeoQuery Dataset** | Benchmark data for training and evaluation |

### Key PyTorch Components
- `nn.Embedding` — learned word embeddings for input and output vocabularies
- `nn.LSTM` — recurrent encoder and decoder layers
- `nn.utils.rnn.pack_padded_sequence` / `pad_packed_sequence` — efficient batched RNN computation
- `nn.Linear` — output projection from hidden state to vocabulary logits
- `nn.LogSoftmax` + `NLLLoss` — standard classification loss for sequence generation
- `optim.Adam` — gradient-based optimizer

## Project Structure

```
.
├── main.py           # Entry point: argument parsing, data loading, training, evaluation
├── models.py         # NearestNeighborSemanticParser, Seq2SeqSemanticParser, encoder/decoder modules
├── data.py           # Dataset loading, tokenization, vocabulary indexing, Example/Derivation classes
├── utils.py          # Indexer (bijective word↔int mapping), Beam search data structure
├── lf_evaluator.py   # Evaluation harness: calls Java backend, computes denotation accuracy
├── geo_test_output.tsv  # Model predictions on the blind test set
└── data/
    ├── geo_train.tsv
    ├── geo_dev.tsv
    └── geo_test.tsv
```

## Usage

**Train and evaluate the seq2seq model:**
```bash
python main.py
```

**Run the nearest neighbor baseline:**
```bash
python main.py --do_nearest_neighbor
```

**Skip Java denotation evaluation (faster, no Java backend required):**
```bash
python main.py --no_java_eval
```

**Print dataset samples after loading:**
```bash
python main.py --print_dataset
```

**Key hyperparameters (set in `models.py` / `train_model_encdec`):**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--decoder_len_limit` | 65 | Max output sequence length |
| `--seed` | 0 | Random seed for reproducibility |

Embedding dimension is fixed at 256 and LSTM hidden size at 400 in the current implementation.

## Background

This project is based on the GeoQuery semantic parsing task introduced by Zelle & Mooney (1996) and widely used as a benchmark for natural language interfaces to databases. The logical form preprocessing (De Bruijn variable standardization) follows the approach of Jia & Liang (2016).
