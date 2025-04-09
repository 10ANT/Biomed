def modality_targets_from_target_dist(target_dist: dict) -> dict:
    return {modality: list(targets.keys()) for modality, targets in target_dist.items()}
