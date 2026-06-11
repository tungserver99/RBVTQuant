# RBVTQuant

`RBVTQuant` is a separate implementation of the soft-relaxation assignment method
described in `RBVT_soft_relaxation_note.md`.

Design choices:

- Keep the block-wise codebook/scaling backbone from `NCCQuant`.
- Support both plain nearest-codeword quantization (`RTN`) and `RBVT`.
- Use one unified entrypoint for quantization and perplexity evaluation.

Main entrypoint:

```bash
cd RBVTQuant
python main.py \
  --model-path <hf-model-or-local-path> \
  --method rbvt \
  --quantizer nf4 \
  --output-dir ./rbvt_model
```

Run plain RTN with the same backbone:

```bash
cd RBVTQuant
python main.py \
  --model-path <hf-model-or-local-path> \
  --method rtn \
  --quantizer nf4 \
  --output-dir ./rtn_model
```
