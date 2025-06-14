import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import torch
import splice
from PIL import Image
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go

@foo.operator
class DecomposeDatasetConcepts(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="decompose_dataset_concepts",
            label="Decompose Dataset Concepts",
            description="Decompose dataset-level concepts using SpLiCE",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Add model selection
        model_choices = types.Choices()
        model_choices.add_choice("open_clip:ViT-B-32", label="OpenCLIP ViT-B-32")
        model_choices.add_choice("clip:ViT-B/32", label="CLIP ViT-B/32")
        model_choices.add_choice("clip:ViT-B/16", label="CLIP ViT-B/16")
        model_choices.add_choice("clip:RN50", label="CLIP RN50")
        
        inputs.enum(
            "model",
            model_choices.values(),
            default="open_clip:ViT-B-32",
            label="Model",
            description="Select the CLIP backbone to use",
        )
        
        # Add vocabulary selection
        vocab_choices = types.Choices()
        vocab_choices.add_choice("laion", label="LAION")
        vocab_choices.add_choice("mscoco", label="MSCOCO")
        vocab_choices.add_choice("laion_bigrams", label="LAION Bigrams")
        
        inputs.enum(
            "vocabulary",
            vocab_choices.values(),
            default="laion",
            label="Vocabulary",
            description="Select the concept vocabulary to use",
        )
        
        # Add vocabulary size
        inputs.int(
            "vocab_size",
            default=10000,
            label="Vocabulary Size",
            description="Number of concepts to consider (top-k most frequent)",
        )
        
        # Add L1 penalty
        inputs.float(
            "l1_penalty",
            default=0.25,
            label="L1 Penalty",
            description="Controls sparsity of the decomposition (higher = sparser)",
        )
        
        # Add number of top concepts to show
        inputs.int(
            "top_k",
            default=20,
            label="Top K Concepts",
            description="Number of top concepts to show",
        )
        
        # Add batch size
        inputs.int(
            "batch_size",
            default=32,
            label="Batch Size",
            description="Number of samples to process at once",
        )
        
        return types.Property(inputs)

    def execute(self, ctx):
        # Get parameters
        model_name = ctx.params.get("model", "open_clip:ViT-B-32")
        vocabulary = ctx.params.get("vocabulary", "laion")
        vocab_size = ctx.params.get("vocab_size", 10000)
        l1_penalty = ctx.params.get("l1_penalty", 0.25)
        top_k = ctx.params.get("top_k", 20)
        batch_size = ctx.params.get("batch_size", 32)
        
        # Get selected samples
        samples = ctx.selected
        if not samples:
            return {"error": "No samples selected"}
        
        # Initialize SpLiCE model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        splicemodel = splice.load(
            model_name,
            vocabulary,
            vocab_size,
            device,
            l1_penalty=l1_penalty,
            return_weights=True
        )
        
        # Get preprocessing
        preprocess = splice.get_preprocess(model_name)
        
        # Get vocabulary
        vocab = splice.get_vocabulary(vocabulary, vocab_size)
        
        # Initialize dataset-level statistics
        total_samples = len(samples)
        concept_weights = torch.zeros(vocab_size, device=device)
        concept_counts = torch.zeros(vocab_size, device=device)
        concept_weight_sums = torch.zeros(vocab_size, device=device)
        concept_weight_sums_sq = torch.zeros(vocab_size, device=device)
        
        # Process samples in batches
        for i in range(0, total_samples, batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # Process each sample in the batch
            for sample in batch_samples:
                # Load and preprocess image
                img = Image.open(sample.filepath)
                img_tensor = preprocess(img).to(device).unsqueeze(0)
                
                # Get concept decomposition
                weights, l0_norm, cosine = splice.decompose_image(img_tensor, splicemodel, device)
                weights = weights.squeeze()
                
                # Update statistics
                concept_weights += weights
                concept_counts += (weights > 0).float()
                concept_weight_sums += weights
                concept_weight_sums_sq += weights * weights
                
                # Store sample-level results
                _, indices = torch.sort(weights, descending=True)
                top_concepts = []
                for idx in indices[:top_k]:
                    if weights[idx.item()].item() > 0:
                        concept = vocab[idx.item()]
                        weight = weights[idx.item()].item()
                        top_concepts.append({
                            "concept": concept,
                            "weight": weight
                        })
                
                sample["splice_concepts"] = top_concepts
                sample["splice_l0_norm"] = l0_norm
                sample["splice_cosine_sim"] = cosine
                sample.save()
        
        # Compute final statistics
        concept_weights /= total_samples
        concept_means = concept_weight_sums / total_samples
        concept_vars = (concept_weight_sums_sq / total_samples) - (concept_means * concept_means)
        concept_stds = torch.sqrt(torch.clamp(concept_vars, min=0))
        
        # Get top-k concepts
        _, indices = torch.sort(concept_weights, descending=True)
        top_concepts = []
        for idx in indices[:top_k]:
            if concept_weights[idx.item()].item() > 0:
                concept = vocab[idx.item()]
                weight = concept_weights[idx.item()].item()
                count = concept_counts[idx.item()].item()
                mean_weight = concept_means[idx.item()].item()
                std_weight = concept_stds[idx.item()].item()
                
                top_concepts.append({
                    "concept": concept,
                    "weight": weight,
                    "count": int(count),
                    "mean_weight": mean_weight,
                    "std_weight": std_weight
                })
        
        # Store dataset-level results
        dataset = ctx.dataset
        dataset.dataset_splice_concepts = {
            "concepts": top_concepts,
            "total_samples": total_samples
        }
        dataset.save()
        
        return {"message": f"Processed {total_samples} samples"}

def create_dataset_visualizations(dataset):
    """Create and store dataset-level visualizations."""
    if not dataset.has_field("dataset_splice_concepts"):
        return
    
    concepts_data = dataset.dataset_splice_concepts
    if not concepts_data:
        return
    
    # Create concept frequency plot
    concepts = [c["concept"] for c in concepts_data["concepts"]]
    counts = [c["count"] for c in concepts_data["concepts"]]
    
    fig_freq = go.Figure(data=go.Bar(
        x=concepts,
        y=counts,
        text=[f"{c}" for c in counts],
        textposition='auto',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig_freq.update_layout(
        title="Concept Frequency in Dataset",
        xaxis_title="Concept",
        yaxis_title="Number of Samples",
        height=400,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        )
    )
    
    # Create concept weight distribution plot
    weights = [c["weight"] for c in concepts_data["concepts"]]
    
    fig_weights = go.Figure(data=go.Bar(
        x=concepts,
        y=weights,
        text=[f"{w:.3f}" for w in weights],
        textposition='auto',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig_weights.update_layout(
        title="Average Concept Weights in Dataset",
        xaxis_title="Concept",
        yaxis_title="Average Weight",
        height=400,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        )
    )
    
    # Store visualizations in dataset
    if not dataset.has_field("dataset_splice_visualizations"):
        dataset.add_field("dataset_splice_visualizations", fo.DictField)
    
    dataset.dataset_splice_visualizations = {
        "concept_frequency": fig_freq.to_json(),
        "concept_weights": fig_weights.to_json()
    }
    dataset.save()

def register(p):
    p.register(DecomposeDatasetConcepts)