"""Ignore Colab's optional audio/vision packages for this text-only model."""
import os
os.environ.update(USE_TORCH='1',USE_TF='0',USE_FLAX='0')
import transformers.utils.import_utils as optional
optional._librosa_available=False
optional._torchvision_available=False
