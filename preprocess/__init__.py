"""Preprocessing primitives.

    - preprocess.filtering : 4-40 Hz zero-phase Butterworth bandpass
    - preprocess.windows   : 2-s sliding-window epoching with 1-s stride
                             and the WindowedDataset container consumed
                             by every model and attack module
"""
