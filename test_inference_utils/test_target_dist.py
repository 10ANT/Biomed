from inference_utils.target_dist import modality_targets_from_target_dist


def test_modality_targets_from_target_dist():
    target_dist_dict = {
        "CT-Abdomen": {"postcava": [[1, 2], [3, 4]], "aorta": [[5, 6], [7, 8]]},
        "CT-Chest": {"nodule": [[9, 10], [11, 12]], "tumor": [[13, 14], [15, 16]]},
    }

    expected = {
        "CT-Abdomen": ["postcava", "aorta"],
        "CT-Chest": ["nodule", "tumor"],
    }

    actual = modality_targets_from_target_dist(target_dist_dict)

    assert actual == expected
