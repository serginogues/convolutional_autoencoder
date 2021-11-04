"""
Convolutional Autoencoder

1) Goal: Reconstruct any CIFAR-10 image
- Cost function: MSE or binary cross-entropy between input and reconstructed image
- Activation for conv layers: ReLU
- Architecture:
    Input: 32x32x3 (colored image)
    Encoder:
        Layer 1: Filter: 3x3 and 8 channels -> Max pooling 2x2
        Layer 2: Filter: 3x3 + Channels: 12 -> Max pooling 2x2
    Decoder:
        Layer 3: Filter: 3x3 and 16 channels -> Upsampling 2x2
        Layer 4: Filter: 3x3 and 12 channels -> Upsampling 2x2
        Layer 5: Filter: 3x3 and 3 channels
"""