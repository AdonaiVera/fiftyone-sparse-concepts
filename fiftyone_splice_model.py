import torch
import numpy as np

from .splice import (
    load,
    get_preprocess,
    get_vocabulary,
    decompose_image
)

import fiftyone.core.labels as fol
import fiftyone.utils.torch as fout

class SpliceModelConfig(fout.TorchImageModelConfig):

    def __init__(self, d):
        super().__init__(d)

        self.model_name = self.parse_string(
            d, "model_name", default="open_clip:ViT-B-32"
        )
        self.vocabulary_name = self.parse_string(
            d, "vocabulary_name", default="laion"
        )
        self.vocabulary_size = self.parse_int(
            d, "vocabulary_size", default=10000
        )
        self.l1_penalty = self.parse_number(
            d, "l1_penalty", default=0.25
        )
        self.top_k = self.parse_int(
            d, "top_k", default=128
        )
        self.return_cosine = self.parse_bool(
            d, "return_cosine", default=False
        )
        self.save_l0_norm = self.parse_bool(
            d, "save_l0_norm", default=False
        )

        self.entrypoint_fcn = load

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.entrypoint_args = {
            "name": self.model_name,
            "vocabulary": self.vocabulary_name,
            "vocabulary_size": self.vocabulary_size,
            "device": device,
            "l1_penalty": self.l1_penalty,
            "return_weights": True,
            "return_cosine": self.return_cosine,
        }

        self.transforms_fcn = get_preprocess
        self.transforms_args = {
            "name": self.model_name,
        }

        self.ragged_batches = False

        self.output_processor_cls = ConceptPostProcessor
        self.output_processor_args = {
            "vocabulary_name": self.vocabulary_name,
            "vocabulary_size": self.vocabulary_size,
            "top_k": self.top_k,
            "save_l0_norm": self.save_l0_norm,
            "save_reconstruction_error": self.return_cosine,
        }


class SpliceModel(fout.TorchImageModel):

    def __init__(self, config):
        super().__init__(config)


    def _forward_pass(self, imgs):
        return decompose_image(
            imgs,
            self._model,
            self.device,
            return_cosine=self.config.return_cosine,
        )

class ConceptPostProcessor:

    def __init__(
        self,
        vocabulary_name,
        vocabulary_size,
        top_k,
        classes=None,
        save_l0_norm=False,
        save_reconstruction_error=False,
    ):
        self.vocabulary_name = vocabulary_name
        self.vocabulary = np.array(
            get_vocabulary(vocabulary_name, vocabulary_size)
        )
        self.top_k = top_k

        self.save_l0_norm = save_l0_norm
        self.save_reconstruction_error = save_reconstruction_error

    def __call__(self, output, frame_size, confidence_thresh=None):

        if self.save_reconstruction_error:
            weights, cosine = output
            cosine = cosine.detach().cpu().numpy()
        else:
            weights = output

        batch_size = weights.shape[0]

        _, indices = torch.sort(weights.squeeze(), descending=True)

        indicies = indices[:, :self.top_k].detach().cpu().numpy()

        concepts = self.vocabulary[indicies]

        if self.save_l0_norm:
            l0_norms = torch.sum(weights > 0, dim=-1).detach().cpu().numpy()

        weights = weights.detach().cpu().numpy()

        labels = []
        for i in range(batch_size):

            label = fol.Classifications(
                classifications = [
                    fol.Classification(
                        label=concepts[i, j],
                        weight=weights[i, j],
                    ) for j in range(self.top_k)
                ],
                l0_norm=None if not self.save_l0_norm else l0_norms[i],
                reconstruction_error=None if not self.save_reconstruction_error else cosine[i],
            )

            labels.append(label)
        
        return labels







    
        