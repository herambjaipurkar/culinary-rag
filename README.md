# 🍛 South Asian Culinary RAG Assistant

An intelligent, completely local Retrieval-Augmented Generation (RAG) chatbot designed exclusively for South Asian cuisine. Built with LangGraph, FAISS, and Qwen-0.5B, this assistant acts as your personal sous-chef—remembering your ingredients across chat turns, performing mathematically precise semantic recipe searches, and streaming beautifully formatted Markdown responses through a premium, ChatGPT-style interface.

---

## ✨ Key Features

### 🎨 Premium UI Experience (app.py)

- **ChatGPT-Style Interface:** A completely custom CSS dark-mode UI with right-aligned user pill bubbles, subtle action buttons, and a clean charcoal aesthetic.  
- **Smart Session Management:** Multi-chat sidebar functionality allowing you to easily switch between different ongoing recipe conversations.  
- **Interactive Controls:** Seamlessly edit previous messages or "Retry" generation with a single click.  
- **"Under the Hood" Data Expander:** Click the 🔍 View Pipeline Data expander on any response to instantly see the AI's internal state: Active Intent, Extracted Ingredients, Raw Chunks Used, and exact FAISS L2 Scores.  
- **Flawless Streaming:** Regex-powered real-time typing effect that perfectly preserves complex Markdown bullet points and bolding.  

---

### 📚 Intelligent Vector Database (vector_db_setup.py)

- **Structured Chunking:** Splits monolithic recipes into 3 distinct, metadata-rich chunks (Introduction, Ingredients, Instructions) to maximize semantic retrieval accuracy.  
- **Dense Embeddings:** Utilizes BAAI/bge-large-en-v1.5 with normalize_embeddings=True to create highly stable 1024-dimensional mathematical vectors.  
- **L2 Distance Thresholding:** Strict <= 0.55 distance cutoff using normalized cosine-similarity embeddings to ensure the LLM only receives highly accurate context.  

---

### 🧠 Under the Hood: LangGraph Architecture

At its core, this assistant uses LangGraph to model the AI as a state machine (assistant_core.py). Think of it as a flowchart where a "backpack" of data travels between different specialized stations.

#### 1. The "Backpack" (GraphState)

Data travels through the pipeline inside a shared dictionary called the State. As the user chats, the backpack dynamically updates with memory and context:

- **question:** The user's input.  
- **chat_history:** Past messages for conversational memory.  
- **intent:** The detected user goal (e.g., DISH_QUERY).  
- **extracted:** Active variables (e.g., {'ingredients': ['egg', 'rice']}).  
- **raw_chunks:** The recipe text and exact FAISS L2 scores.  

---

#### 2. The "Stations" (Nodes)

The backpack travels to specific Python functions (Nodes) that perform dedicated tasks:

- 🕵️‍♂️ **The Detective (classify_intent_node):** Reads the chat history and extracts active ingredients/preferences using regex. It assigns the master intent and catches conversational edge cases (like saying "yes" to a South Asian alternative).  

- 📚 **The Librarian (retrieve_node):** Strips generic filler words and builds a highly concentrated semantic query. It fetches up to 15 chunks from FAISS, groups them, and runs the Hybrid Scorer to guarantee the most relevant dish wins over semantic anomalies (e.g., preventing "Eggplant" from hijacking searches for "Egg").  

- 👨‍🍳 **The Chef (generate_recipe_node):** Injects the raw, messy chunks of the winning recipe into a strict System Prompt and uses Qwen-0.5B to rewrite it into beautiful, standardized Markdown.  

- 🚨 **The Fallbacks:** Nodes like clarify_ingredients_node or out_of_bounds_node that seamlessly handle vague or non-South-Asian requests.  

---

#### 3. The "Traffic Cop" (Conditional Routing)

After the Detective reviews the input, the Traffic Cop (route_logic) takes over. If the request is casual or out of bounds, it short-circuits the retrieval process and routes the backpack directly to a fallback node. If it's a valid recipe request, it sends the backpack straight to the Librarian.

---

## 🧩 System Architecture

```text
                        ┌──────────────────────────┐
                        │       User Input         │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │  Streamlit UI (app.py)   │
                        │  - Chat Interface        │
                        │  - Session State         │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   LangGraph Controller   │
                        │   (assistant_core.py)    │
                        └────────────┬─────────────┘
                                     │
                     ┌───────────────┼────────────────┐
                     │               │                │
                     ▼               ▼                ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐
        │ Intent Classifier │  │   Route Logic    │  │   Chat History     │
        │ (Detective Node)  │  │ (Traffic Cop)    │  │   (State Memory)   │
        └─────────┬────────┘  └─────────┬────────┘  └─────────┬──────────┘
                  │                     │                     │
                  │                     ▼                     │
                  │        ┌──────────────────────────┐       │
                  │        │   Retrieval Triggered?   │◄──────┘
                  │        └────────────┬─────────────┘
                  │                     │
         YES ─────┘                     └────── NO (Fallback)
                  │                             │
                  ▼                             ▼
        ┌──────────────────┐         ┌────────────────────────┐
        │  FAISS Retriever │         │  Fallback Nodes        │
        │  (Librarian)     │         │  - Clarification       │
        │  - Top K Chunks  │         │  - Out-of-Bounds       │
        │  - L2 Filtering  │         └────────────────────────┘
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────────────────┐
        │ Hybrid Scoring + Grouping    │
        │ - Chunk Aggregation          │
        │ - Keyword Boost              │
        └─────────┬────────────────────┘
                  │
                  ▼
        ┌──────────────────────────────┐
        │   LLM Generation (Qwen)      │
        │   (Chef Node)                │
        │ - Context Injection         │
        │ - Markdown Formatting       │
        └─────────┬────────────────────┘
                  │
                  ▼
        ┌──────────────────────────────┐
        │  Streaming Response Output   │
        │  (Formatted Markdown)        │
        └──────────────────────────────┘
```

## 🚀 Installation & Setup

### 1. Prerequisites

You will need Python 3.9+ and a machine with at least 4GB to 8GB of free RAM/VRAM to run the local LLM and embedding models smoothly. (Apple Silicon Macs are fully supported via MPS).

---

### 2. Clone the Repository

```bash
git clone https://github.com/Mithil21/culinary_rag_assistant.git
cd culinary_rag_assistant
```

---

### 3. Set Up a Virtual Environment (Highly Recommended)

```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate  

# On Windows:
venv\Scripts\activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Build the Vector Database

Ensure your raw corpus JSON (vector_ready_corpus_with_metadata.json) is in the root directory. Run the indexing script to generate the FAISS semantic embeddings:

```bash
python vector_db_setup.py
```

(This will generate a ./faiss_index/ folder containing index.faiss and index.pkl)

---

### 6. Launch the Assistant

Boot up the Streamlit interface:

```bash
streamlit run app.py
```

---

## 💡 Usage Examples

### Example 1: Direct Exact Match

**User:** "How do I make Thalassery Biryani?"  

**Assistant:** Bypasses clarification, instantly retrieves the specific recipe, and streams the formatted instructions.

---

### Example 2: Dynamic Ingredient Memory

**User:** "I have eggs and rice."  

**Assistant:** "I can work with these ingredients: egg, rice. Do you want a quick South Asian meal, a curry, or something rice-based?"  

**User:** "Rice based."  

**Assistant:** Reads the LangGraph state, concatenates 'egg' and 'rice' into the search query, triggers the Keyword Bonus to beat 'Eggplant', and outputs 'Egg Rice'.

---

### Example 3: Out of Bounds Handling

**User:** "Give me a recipe for Pizza."  

**Assistant:** "My database is focused on South Asian cuisine only. If you want, I can still suggest a similar South Asian alternative."  

**User:** "Yes."  

**Assistant:** Dynamically intercepts the "yes", maps 'Pizza' to 'Naan/Uttapam/Flatbread', and searches the vector database for the South Asian equivalent.

---

## 📊 Benchmark Results

The system was evaluated on 100 diverse queries spanning dish lookups, ingredient-based searches, vague inputs, and out-of-scope requests.

### 🔢 Overall Metrics

- **Total Queries:** 100  
- **Mean Recall@3:** 0.9241  
- **Intent Classification Accuracy:** 0.93  
- **Generation Faithfulness:** 0.8861  

---

### ⚡ Latency Performance

- **Mean Latency:** 7.528 seconds  
- **Min Latency:** 0.0 seconds  
- **Max Latency:** 18.608 seconds  

---

### 🧠 Intent Classification Breakdown

| Intent Type            | Accuracy | Correct | Total |
|----------------------|----------|--------|-------|
| ALTERNATIVE_REQUEST  | 1.0      | 1      | 1     |
| DISH_QUERY           | 0.9      | 36     | 40    |
| INGREDIENTS_ONLY     | 1.0      | 11     | 11    |
| NON_SOUTH_ASIAN      | 1.0      | 1      | 1     |
| RECIPE_REQUEST       | 0.9268   | 38     | 41    |
| SUGGESTION_REQUEST   | 1.0      | 5      | 5     |
| VAGUE_REQUEST        | 1.0      | 1      | 1     |

---

### 📌 Key Observations

- Strong **retrieval quality** with Recall@3 > 92%  
- Near-perfect handling of **edge-case intents** (vague, alternative, out-of-scope)  
- Slight drop in **dish-specific queries**, indicating room for improved disambiguation  
- **Latency variance** driven primarily by local LLM inference time  

---



## 🛠️ Hardware Notes & Optimizations

- **Apple Silicon (M1/M2/M3):** The assistant_core.py script automatically detects macOS architecture and utilizes mps (Metal Performance Shaders) with torch.float16 precision for ultra-fast local inference.  

- **NVIDIA GPUs:** Automatically detects CUDA and utilizes float16 VRAM acceleration.  

- **CPU Only:** Defaults to standard float32 for mathematical stability on machines without dedicated AI accelerators. Pipeline enforces do_sample=False for deterministic, crash-free generation across all hardware.