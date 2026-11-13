"""
Phase 4 Tests: Deep Learning Model Architectures (Static & Temporal)
Verifies Custom CNN, MobileNetV3, ResNet18, Bi-GRU, LSTM, Transformer, and Temporal Buffer.
"""

import os
import sys
import unittest
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.classification.models import ISLCustomCNN, ISLMobileNetV3, ISLResNet18, build_model
from ml.temporal.temporal_models import ISLBiGRU, ISLLSTM, ISLTransformer, TemporalSequenceBuffer


class TestPhase4(unittest.TestCase):
    def test_custom_cnn(self):
        """Verify Custom CNN forward pass and feature extraction."""
        model = ISLCustomCNN(num_classes=26)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

        feat = model.extract_features(x)
        self.assertEqual(feat.dim(), 2)
        self.assertEqual(feat.size(0), 2)

    def test_mobilenet_v3(self):
        """Verify MobileNetV3-Small forward pass and output dimensions."""
        model = ISLMobileNetV3(num_classes=26, pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

    def test_resnet18(self):
        """Verify ResNet-18 forward pass and output dimensions."""
        model = ISLResNet18(num_classes=26, pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 26))

    def test_bi_gru(self):
        """Verify Bi-GRU temporal sequence recognition."""
        model = ISLBiGRU(input_dim=256, hidden_dim=64, num_classes=26)
        # Batch: 2, Sequence Length: 16, Feature Dim: 256
        seq = torch.randn(2, 16, 256)
        out = model(seq)
        self.assertEqual(out.shape, (2, 26))

    def test_lstm(self):
        """Verify Multi-layer LSTM temporal gesture recognition."""
        model = ISLLSTM(input_dim=256, hidden_dim=64, num_classes=26)
        seq = torch.randn(2, 16, 256)
        out = model(seq)
        self.assertEqual(out.shape, (2, 26))

    def test_transformer(self):
        """Verify Transformer Encoder self-attention sequence modeling."""
        model = ISLTransformer(input_dim=256, d_model=64, nhead=4, num_classes=26)
        seq = torch.randn(2, 16, 256)
        out = model(seq)
        self.assertEqual(out.shape, (2, 26))

    def test_temporal_sequence_buffer(self):
        """Verify sliding window buffer fills and returns (1, 16, 256) sequence."""
        buf = TemporalSequenceBuffer(sequence_length=16, feature_dim=256)
        self.assertFalse(buf.is_ready())
        self.assertIsNone(buf.get_sequence_tensor())

        # Feed 15 frames: should still not be ready
        for _ in range(15):
            buf.add_frame_features(torch.randn(256))
        self.assertFalse(buf.is_ready())

        # 16th frame: buffer becomes ready
        buf.add_frame_features(torch.randn(256))
        self.assertTrue(buf.is_ready())
        tensor = buf.get_sequence_tensor()
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 16, 256))


if __name__ == "__main__":
    unittest.main()

