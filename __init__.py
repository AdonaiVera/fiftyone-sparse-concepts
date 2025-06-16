import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

from .decompose_core_operator import DecomposeCoreConcepts
from .decompose_core_panel import DecomposeCorePanel

def register(plugin):
    """Register operators with the plugin."""
    plugin.register(DecomposeCoreConcepts)
    plugin.register(DecomposeCorePanel) 