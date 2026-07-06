"""Optional speech ML helpers for EarlyCare research workflows.

The modules in this package avoid importing heavyweight ML dependencies at app
startup. Training and prediction scripts import tabular ML, Parselmouth, and
other optional libraries only when those workflows are explicitly run.
"""

TARGET_SAMPLE_RATE = 16_000
