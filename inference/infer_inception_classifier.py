#!/usr/bin/env python3
"""Compatibility entrypoint for the InceptionV3 static image classifier.

This keeps the new script name aligned with the actual model family while
reusing the implementation in `infer_vgg_classifier.py`.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.infer_vgg_classifier import main


if __name__ == "__main__":
    main()