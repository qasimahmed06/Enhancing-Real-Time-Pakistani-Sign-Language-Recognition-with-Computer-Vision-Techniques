"""Compatibility wrapper for the VGG feature extractor.

The project currently defines `VGGFeatureExtractor` in
`models.inception_feature_extraction`, but several modules import it from
`models.vgg_feature_extractor`. This module keeps that public import path
available without duplicating the implementation.
"""

from models.inception_feature_extraction import VGGFeatureExtractor
