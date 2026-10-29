# Vehicle Dynamics RAG

## About

Vehicle Dynamics RAG is a command-line question answering app for one source: the Wikipedia article on [vehicle dynamics](https://en.wikipedia.org/wiki/Vehicle_dynamics). You type a question about handling, tires, simulation, or the other topics in that article, and the app answers from the article text stored in `Selected_Document.txt`.

The article is a compact, sectioned reference. It covers the study of how a vehicle’s motion changes with driver inputs, propulsion, and the road or air around it, and it groups the design factors that drive that motion: drivetrain and braking, suspension and steering, mass distribution, aerodynamics, and tires. Those sections are distinct enough that a retriever can pull a relevant passage instead of sending the whole page to the model.

Answers stay tied to that page. A local index finds candidate passages, a cross-encoder re-ranks them, and GPT-4o writes the reply from the passages it receives. When the article does not contain the answer, the prompt tells the model to say so. Chunk size and overlap are constants at the top of `src/RAG_app.py`, so the same question can be asked again after those settings change. The chunk-size experiments later in this file record that comparison.

## How it works

1. `src/text_extractor.py` downloads the Wikipedia page and keeps the paragraphs and lists inside `mw-content-text`.
2. `src/RAG_app.py` splits that text into overlapping chunks.
3. `sentence-transformers/all-distilroberta-v1` embeds every chunk.
4. A FAISS `IndexFlatL2` index stores those vectors and returns the nearest chunks by squared Euclidean distance. That index is an exact search, which is a good fit for one article.
5. `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each retrieved chunk together with the question and keeps the best ones. The bi-encoder is fast enough to scan the index; the cross-encoder is the second pass that checks query and chunk jointly.
6. GPT-4o (`temperature=0`) answers from those chunks. The system prompt tells it to say it does not know when the context does not contain the answer.

Overlap matters because a split can cut a sentence in half. Shared characters on both sides of a boundary keep a concept that straddles the cut inside at least one chunk.

## Setup

Python 3.10 or newer. The first run downloads the two Hugging Face models, so it needs network access.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is gitignored):

```
OPENAI_API_KEY=your_key_here
```

The key needs access to `gpt-4o`. Each question is one chat completion.

## Run

Run both scripts from the project root. They read and write `Selected_Document.txt` relative to the current directory.

Refresh the corpus (optional; the extracted article is already in the repo):

```bash
python src/text_extractor.py
```

Start the question loop:

```bash
python src/RAG_app.py
```

Type a question at the `Your question:` prompt. Enter `exit` or `quit` to stop.

Example questions the article can support:

- What role do tires play in vehicle dynamics simulation?
- How do simulations aid in vehicle dynamics?
- Explain the difference between oversteer and understeer.

On a question the article does not cover, the model is instructed to say it does not know rather than fill the gap from pretraining.

## Configuration

Defaults are at the top of `src/RAG_app.py`.

| Setting | Default | Role |
| --- | --- | --- |
| `chunk_size` | 500 | Maximum characters per chunk |
| `chunk_overlap` | 50 | Characters repeated across neighboring chunks |
| `model_name` | `sentence-transformers/all-distilroberta-v1` | Bi-encoder used for the FAISS vectors |
| `top_k` | 20 | Chunks retrieved from FAISS |
| `cross_encoder_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranker |
| `top_m` | 8 | Chunks passed to the language model after re-ranking |
| Chat model | `gpt-4o` | Answer model, `temperature=0`, `max_tokens=500` |

Embedding dimensionality is the length of each `SentenceTransformer` vector. A longer vector can hold finer semantic detail, and it also raises the memory and search cost of the FAISS index.

## Chunk-size experiments

The same question — “How do simulations aid in vehicle dynamics?” — was run at three splitter settings. The article has a section on simulation, so a refusal means the chunks were too small for retrieval to assemble that section.

| Chunk size | Overlap | Result |
| --- | --- | --- |
| 1000 | 500 | Correct, shorter answer: simulations evaluate many conditions faster than real time, model tire-road shear forces, and support software-in-the-loop and hardware-in-the-loop controller tests (CarSim and similar tools). |
| 500 | 100 | Correct and more detailed: same simulation role, plus how vehicle motion responds to driver inputs, propulsion, and the environment. |
| 100 | 50 | “I don’t know.” The passage was split too finely for the retriever to return a usable context. |

Mid-size chunks (500 / 100) gave the fullest answer. The largest setting was accurate and more compact. The smallest setting dropped the answer entirely.

## Project layout

```
.
├── README.md
├── requirements.txt
├── Selected_Document.txt    # extracted Wikipedia article
└── src
    ├── text_extractor.py    # fetch and save the article
    └── RAG_app.py           # index, retrieve, re-rank, answer
```
