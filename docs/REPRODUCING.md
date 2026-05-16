# Reproducing and Auditing the Headline Numbers

CryoSentinel v1.0 is a model-first public release. The released checkpoint, dataset, benchmark tables, per-chip diagnostics, threshold sweeps, figures, and label-noise audit are public. The private cloud orchestration and internal training scripts used to produce the v1.0 run are not included in this repository.

## Public Artifacts

- Model repository: `abzal-glw/cryosentinel-terramind-v3`
- Dataset repository: `abzal-glw/cryosentinel-glof-v3`
- Production checkpoint: `terramind_v3_finetune_almaty_v2/checkpoints/soup.ckpt`
- Single-best checkpoint: `terramind_v3_finetune_almaty_v2/checkpoints/step001605-iou0.952.ckpt`
- Public diagnostics: `terramind_v3_finetune_almaty_v2__soup_tta1/diagnostics/`
- Dataset statistics: see `docs/DATA.md`

## Download the Checkpoint

```bash
huggingface-cli download \
    abzal-glw/cryosentinel-terramind-v3 \
    terramind_v3_finetune_almaty_v2/checkpoints/soup.ckpt \
    --local-dir ./checkpoints
```

## Download the Dataset

```bash
huggingface-cli download \
    --repo-type dataset \
    abzal-glw/cryosentinel-glof-v3 \
    --local-dir ./data/multimodal_chips_v3
```

## Expected Headline Metrics

The released benchmark table should match:

```text
val global IoU @ thr=0.5, TTA                 : 0.9557
val global IoU @ best thr=0.70, TTA           : 0.9596
test global IoU @ thr=0.5, TTA                : 0.8918
test global IoU @ thr=0.5, label-corrected    : 0.9082

per-region test IoU @ thr=0.5:
  ile_alatau                                  : 0.7664
  tien_shan_full                              : 0.9027
  zhetysu_alatau                              : 0.9312

per-region test IoU @ thr=0.5, label-corrected:
  ile_alatau                                  : 0.9285
  tien_shan_full                              : 0.9027
  zhetysu_alatau                              : 0.9312
```

## What Is Not Public in v1.0

The v1.0 public repository does not include the internal cloud runners, checkpoint-recovery scripts, batch diagnostic jobs, or data-ingestion orchestration used during development. Those components belong to the private development and operational stack.

For formal research review, public-sector due diligence, or operational integration discussions, contact the author through GitHub.
