# Selected visual encoder slot

Place your selected **trained** checkpoint here as `best.pt`, or set its actual
filename/path in `encoder.json`. No weights are bundled. Choose the appropriate
configuration from `examples/` after model selection. A checkpoint alone does
not specify arbitrary architecture or preprocessing.

Keep the original training project and preprocessing outputs accessible. The
configuration references them instead of changing those fingerprinted files.
Use only checkpoints and custom Python adapters you trust. Changing this encoder
requires a new embedding cache and retrained fusion head.
