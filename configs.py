"""Named experiment configurations used by training and evaluation."""

PRESETS = {
    "best_empirical": {
        "description": "E8: best empirical topology metrics in this study",
        "padding": 1,
        "dropout": 0.0,
        "use_weights": False,
    },
    "paper_faithful": {
        "description": "E13: paper-like valid convolutions, dropout and weighted loss",
        "padding": 0,
        "dropout": 0.5,
        "use_weights": True,
    },
}

LEARNING_RATE = 1e-3
MOMENTUM = 0.99
BATCH_SIZE = 1
