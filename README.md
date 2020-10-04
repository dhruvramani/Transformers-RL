# Transformers-RL
An easy PyTorch implementation of "Stabilizing Transformers for Reinforcement Learning". I searched around a lot for an easy-to-understand implementation of transformers for RL but couldn't find it. Hence, had to get my hands dirty. 

The stable TransformerXL (GTrXL block in the paper) and other layers are present in `layers.py`. To implement the TrXL-I block, set `gating=False` for `StableTransformerEncoderLayerXL`.

I implemented a basic gausian policy in `policies.py`. Additional policy implementations for different kind of action-spaces are welcome! Send a PR.
