# Sources and compatibility

This package contains newly generated Experiment3 integration code and an
unmodified compatibility copy of the nine Python modules in the Experiment2B
package previously supplied for this project. No external pretrained weights,
LAV-DF recordings, native UMMAFormer source, or AASIST model source are distributed.

PyTorch, NumPy, scikit-learn, tqdm, each selected encoder and its pretraining data,
and LAV-DF retain their respective licenses and terms. The E2B bridges reference
user-installed model projects; they do not override those projects' restrictions.
Only trusted local checkpoint/source files should be used. Restricted
`torch.load(weights_only=True)` is used with no automatic unsafe fallback.

Research references and implementation definitions are listed in METHODS.md.
