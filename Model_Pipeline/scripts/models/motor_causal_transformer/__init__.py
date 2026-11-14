"""The motor causal transformer: a decoder-only transformer over 10-second watch windows.

Owner: P2.

Each window becomes one token (its frozen harnet10 features, plus dose, check-in and
patient fields). Causal attention lets every window see only its own past, positioned by
real time. Gated MIL pooling turns the windows of a diary hour into that hour's prediction:
sleep, waking state (-3..+3) and tremor (0..2), for now and the next hour.

Training has two steps: pretraining on the movement alone (predict the movement 20 s to
2 h ahead), then the diary step (the body frozen, heads and pooling learn the labels).
MODEL.md has the design and why.

A package rather than one file, because the family has several parts. Only model.py is
what the rest of the pipeline imports; the other files are its building blocks.
"""
