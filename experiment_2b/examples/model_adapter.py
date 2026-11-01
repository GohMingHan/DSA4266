"""Explicit interface for a future CNN/YOLO/custom Kaggle model.

Copy into the relevant models/visual or models/audio slot only after implementing
it for the actual trained architecture and EXACT preprocessing. This intentionally
has no pretend fallback model. Existing models may instead export the NPZ/index
contract in README.md and use the preexported adapter.

build_encoder(slot, device) returns an object with:
    records: {exact_metadata_file: input_record}
    provenance: {"representation": "EXACT intermediate layer and preprocessing"}
    info(row): input record including file and split (labels may be audited here)
    input_signature(row): a stable hash of this video's input/processing settings
    encode_inputs(file: str, split: str): dict of arrays described below

Inside encode_inputs:
  model.eval(); model.requires_grad_(False)
  with torch.inference_mode(): extract learned PRE-CLASSIFIER embeddings
  No labels/fake_periods/model probabilities may be embedded into features.

Return for visual:
    features float32 [T,D], timestamp_s float64 [T]
Return for audio:
    features float32 [W,D], start_s/end_s/center_s float64 [W]
All time is in seconds relative to the first decoded VIDEO frame, not audio start.
"""
def build_encoder(slot, device):
    raise NotImplementedError(
        'Custom checkpoint adapter has not been implemented. Supply the actual model '
        'class, preprocessing, and checkpoint keys, or use the preexported interface.'
    )
