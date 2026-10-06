# Example W&B run mappings. Replace the placeholder IDs with your own runs.
ALL_RUN_IDS = {
    "FaceScrub": {
        "baseline": "YOUR_BASELINE_RUN_ID",
        "negls": "YOUR_NEGLS_RUN_ID",
    },
}

# Model names from ALL_RUN_IDS for which to load the last epoch instead of the best.
LOAD_BEST_MODEL_FALSE_LIST = []  # Example: ["negls"]

EVALUATION_MODELS = {
    "FaceScrub": "YOUR_EVALUATOR_RUN_ID",
    "CelebA": "YOUR_CELEBA_EVALUATOR_RUN_ID",
    "CelebACrop": "YOUR_CELEBA_CROP_EVALUATOR_RUN_ID",
    "stanford_dogs": "YOUR_DOGS_EVALUATOR_RUN_ID",
}

# eps is the adversarial-training epsilon used for each robust evaluator.
ROBUST_EVALUATION_MODELS = {
    "FaceScrub": {"run_id": "YOUR_ROBUST_EVALUATOR_RUN_ID", "eps": 0.031},
}