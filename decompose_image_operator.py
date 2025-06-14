import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import torch
import splice
from PIL import Image
import numpy as np

@foo.operator
class DecomposeImageConcepts(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="decompose_image_concepts",
            label="Decompose Image Concepts",
            description="Decompose an image into interpretable concepts using SpLiCE",
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
            default=10,
            label="Top K Concepts",
            description="Number of top concepts to show",
        )
        
        return types.Property(inputs)

    def execute(self, ctx):
        # Get parameters
        model_name = ctx.params.get("model", "open_clip:ViT-B-32")
        vocabulary = ctx.params.get("vocabulary", "laion")
        vocab_size = ctx.params.get("vocab_size", 10000)
        l1_penalty = ctx.params.get("l1_penalty", 0.25)
        top_k = ctx.params.get("top_k", 10)
        
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
        
        # Process each sample
        for sample in samples:
            # Load and preprocess image
            img = Image.open(sample.filepath)
            img_tensor = preprocess(img).to(device).unsqueeze(0)
            
            # Get concept decomposition
            weights, l0_norm, cosine = splice.decompose_image(img_tensor, splicemodel, device)
            
            # Get top-k concepts
            _, indices = torch.sort(weights, descending=True)
            top_concepts = []
            for idx in indices.squeeze()[:top_k]:
                if weights[0, idx.item()].item() > 0:
                    concept = vocab[idx.item()]
                    weight = weights[0, idx.item()].item()
                    top_concepts.append({
                        "concept": concept,
                        "weight": weight
                    })
            
            # Store results in sample
            sample["splice_concepts"] = top_concepts
            sample["splice_l0_norm"] = l0_norm
            sample["splice_cosine_sim"] = cosine
            sample.save()
        
        return {"message": f"Processed {len(samples)} samples"}

def register(p):
    p.register(DecomposeImageConcepts) 