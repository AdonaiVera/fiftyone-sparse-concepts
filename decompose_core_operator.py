import fiftyone.operators as foo
import fiftyone.operators.types as types
import torch
from .fiftyone_splice_model import SpliceModelConfig, SpliceModel 
from PIL import Image
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def clean_memory():
    """Clean up CUDA memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def batch_generator(dataset, batch_size):
    samples = list(dataset.iter_samples())
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


class DecomposeCoreConcepts(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="decompose_core_concepts",
            label="Decompose Core Concepts",
            description="Per-image concept decomposition using SpLiCE (batch-wise, no aggregation)",
            dynamic=True,
        )

    def __call__(self, sample_collection, params=None):
        """Make the operator callable directly.
        
        Args:
            sample_collection: The dataset or view to process
            params: Optional parameters for the operator
            delegate: Whether to delegate execution to the app
        """
        if params is None:
            params = {
                "model": "open_clip:ViT-B-32",
                "vocabulary": "laion",
                "vocab_size": 10000,
                "l1_penalty": 0.25,
                "top_k": 10,
                "batch_size": 32
            }
        ctx = foo.ExecutionContext()
        ctx._dataset = sample_collection
        ctx._params = params
        return self.execute(ctx)

    def resolve_input(self, ctx):
        inputs = types.Object()

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
            description="Select the CLIP backbone to use for concept decomposition"
        )

        vocab_choices = types.Choices()
        vocab_choices.add_choice("laion")
        vocab_choices.add_choice("mscoco")
        vocab_choices.add_choice("laion_bigrams")

        inputs.enum("vocabulary", vocab_choices.values(), default="laion", label="Vocabulary")

        inputs.int("vocab_size", default=10000, label="Vocabulary Size")
        inputs.float("l1_penalty", default=0.25, label="L1 Penalty")
        inputs.int("top_k", default=10, label="Top K Concepts")
        inputs.int(
            "batch_size",
            default=32,
            label="Batch Size",
            description="Number of samples to process at once",
            min=1,
            max=256
        )

        return types.Property(inputs)

    def execute(self, ctx):
        try:
            dataset = ctx.dataset
            if not dataset:
                return {
                    "error": "No dataset found",
                    "description": "Please select a dataset before running the operator."
                }
            
            batch_size = ctx.params.get("batch_size")
            if batch_size <= 0:
                return {
                    "error": "Invalid batch size",
                    "description": "Batch size must be greater than 0."
                }
        
            model_name = ctx.params.get("model")
            vocabulary = ctx.params.get("vocabulary")
            vocab_size = ctx.params.get("vocab_size")
            l1_penalty = ctx.params.get("l1_penalty")
            top_k = ctx.params.get("top_k")

            model_config = SpliceModelConfig(
                {
                    "model_name": model_name,
                    "vocabulary_name": vocabulary,
                    "vocabulary_size": vocab_size,
                    "l1_penalty": l1_penalty,
                    "top_k": top_k,
                    "return_cosine": True,  # TODO: make this configurable
                    "save_l0_norm": True,  # TODO: make this configurable
                }
            )

            splicemodel = SpliceModel(model_config)

            dataset.apply_model(
                splicemodel,
                label_field="concepts", # TODO: make this configurable
                batch_size=batch_size,
            )

            ctx.ops.reload_dataset()
            if hasattr(ctx.ops, "refresh"):
                ctx.ops.refresh()

            logger.info("Concept decomposition completed successfully")

        except Exception as e:
            logger.error(f"Error in execute: {str(e)}")
            return {
                "error": "Operation failed",
                "description": str(e)
            }


def register(p):
    p.register(DecomposeCoreConcepts)
