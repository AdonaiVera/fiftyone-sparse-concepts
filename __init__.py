import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types    

from .decompose_image_operator import DecomposeImageConcepts
from .decompose_class_operator import DecomposeClassConcepts
from .decompose_dataset_operator import DecomposeDatasetConcepts

def register(plugin):
    """Register operators with the plugin."""
    plugin.register(DecomposeImageConcepts)
    plugin.register(DecomposeClassConcepts)
    plugin.register(DecomposeDatasetConcepts)
