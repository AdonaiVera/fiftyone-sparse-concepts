# 🧬 FiftyOne SpLiCE Panel & Operator

A custom **FiftyOne plugin** for exploring and interpreting image-level and dataset-level concept decompositions using **SpLiCE** (Sparse Linear Concept Embeddings). This toolkit helps users visualize, analyze, and detect **spurious correlations** in datasets using interpretable CLIP embeddings.



## 🚀 What This Plugin Offers

### 🔧 Operator: `Decompose Core Concepts`

This operator performs per-image concept decomposition using a CLIP-based model to extract and store the most influential human-interpretable concepts, their contribution weights, and similarity metrics for each image.

#### ✅ Features

* Supports multiple backbones:
  * `open_clip:ViT-B-32` (default)
  * `clip:ViT-B/32`, `ViT-B/16`, `RN50`
* Works with various vocabularies:
  * `laion`, `mscoco`, `laion_bigrams`
* Allows custom tuning:
  * `vocab_size`, `l1_penalty`, `top_k concepts`, `batch_size`
* Stores for each image:
  * `splice_concepts`: list of top contributing concepts with weights
  * `splice_l0_norm`: decomposition sparsity
  * `splice_cosine_sim`: similarity to original CLIP vector

### 📊 Panel: `ImageSplicePanel`

An interactive panel interface with **four analytical views**, powered by the operator’s outputs.

#### 📍 Pages Overview

1. **Dataset-Level Concept Summary**

   * View top x most influential concepts across the dataset
   * Inspect average weight and number of occurrences

2. **Class-Level Decomposition**

   * Select any class and view the top concepts associated with it
   * Helps understand how the model's embeddings represent specific classes

3. **Image-Level Decomposition**

   * When an image is selected, shows its individual concept decomposition
   * Includes a refresh button for up-to-date stats

4. **Spurious Correlation Discovery**

   * Visual tool to detect potential dataset biases
   * Select a concept and observe its distribution across class labels
   * Great for spotting shortcuts or correlations the model may exploit

## 🔌 Installation

### 1. Choose a CLIP backend

* **OpenCLIP**:

```bash
pip install open_clip_torch
```

* **OpenAI CLIP**:

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download the plugin

```bash
fiftyone plugins download https://github.com/AdonaiVera/fiftyone-sparse-concepts
```

---

## 🔍 How to Use

1. **Run the operator** on a view or dataset:

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.operators as foo

# Load the dataset
dataset = foz.load_zoo_dataset("quickstart", max_samples=10)

foo.execute_operator(
    "@adonaivera/fiftyone-sparse-concepts/decompose_core_concepts",
    dataset=dataset,
    view=dataset.view(),
    params={
        "model": "open_clip:ViT-B-32",
        "vocabulary": "laion",
        "vocab_size": 10000,
        "l1_penalty": 0.25,
        "top_k": 10,
        "batch_size": 32,
    },
)

# Launch the app
session = fo.launch_app(dataset)
session.wait()
```

2. **Switch to the panel** `Concept Decomposition`
3. **Explore pages** to:
   * Analyze overall dataset trends
   * Focus on specific classes
   * Inspect individual image decompositions
   * Discover and visualize spurious correlations


## 🔮 Future Functionality

We're actively improving this plugin to support deeper model analysis and broader use cases. Planned enhancements include:

### 🚧 Next Steps

1. **Model Flexibility**
   Adapt the pipeline to support any image encoder or vision-language model beyond CLIP/OpenCLIP.

2. **Multi-Class Support**
   Enable selection and decomposition for multiple class labels simultaneously in class-level views.

3. **Extended Experiment Panels**
   New pages to explore deeper insights and interventions:

   * 📏 **Zero-Shot Accuracy & Cosine Similarity** — Evaluate alignment between concept activations and ground truth.
   * 🧪 **Intervention Studies** — Test how altering concept activations affects model outputs.
   * 🔍 **Retrieval Benchmarks** — Use concept signatures for dataset or sample retrieval.

## 🧠 Based On Research

This plugin implements the decomposition techniques from:

> **SpLiCE: Interpreting CLIP with Sparse Linear Concept Embeddings**
> *Usha Bhalla, Alex Oesterling, Suraj Srinivas, Flavio P. Calmon, Himabindu Lakkaraju*
> [arXiv:2402.10376v2](https://arxiv.org/abs/2402.10376)

Use this to bring **mechanistic interpretability** to your visual embeddings.


## 🙌 Credits

* Built with ❤️ on top of FiftyOne by Voxel51
* SpLiCE paper implementation adapted for batched inference
* Concept visualization powered by `plotly` and custom FiftyOne views

## 👥 Contributors

This plugin was developed and maintained by:

* [Adonai Vera](https://github.com/AdonaiVera) 
* [Jacob Sela](https://github.com/jacobsela) 

We welcome more contributors to extend support for models, vocabularies, and new experiment panels!
