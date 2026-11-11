# PEST-R: placebo-controlled audit of counterfactual removal edits in vision-language models

Code, results and figures for the paper *Do Counterfactual Removal Edits Teach Vision-Language Models a Shortcut? A Placebo-Controlled Audit* (Muhammad, Din, Algarni, Al-Rasheed).

**Question.** Counterfactual training removes an object from an image and changes the answer from Yes to No. Does the model learn "edited image means object absent"? We test this with a *placebo edit*: the same shape and area, placed elsewhere, so the object stays visible and the answer stays Yes. A model that has learned the shortcut loses Yes-answers on placebo images.

## What is in this repository

| Folder | Contents |
|---|---|
| `code/` | Study code (`pest_study.py`, `pest_common.py`, `pest_models.py`), hierarchical bootstrap (`n10_hier_boot.py`), audit builder and analysis (`n11_audit_package.py`, `audit_qa.py`), reference checks (`checks_lib.py`), token scanner (`scrub_tokens.py`), figure script (`make_figs2.py`) |
| `notebooks/` | Kaggle notebooks for the final bootstrap, the statistics audit, the reference checks and the human-audit build |
| `results/` | Final hierarchical-bootstrap output (B = 5000), per-cell and per-seed tables, the trade-off curves, and the human-audit analysis (reviewers appear as A and B only) |
| `figures/` | All figures from the paper as PDF and PNG |

## What is not included, and why

- **Edited images.** The source images are MS COCO images. The COCO Consortium does not own their copyright, and their use follows the Flickr Terms of Use, so we do not redistribute them or edited copies. The scripts regenerate the edits from COCO.
- **Model weights and training checkpoints.** Models keep their original licences (LLaVA-1.5, Qwen2.5-VL, SmolVLM-Instruct). Check each model card before reuse, because some terms restrict commercial use.
- **The audit key and the reviewers' filled forms.** They would reveal which image was the removal edit and who rated it. Only the aggregate analysis is released.

## Running it

The code was run on a free Kaggle T4 GPU (4-bit QLoRA, rank 16). It expects a Kaggle-style working directory (`/kaggle/working`) and a Hugging Face token supplied at run time through Kaggle secrets or `getpass`. **Never put a token in a notebook or script.** `code/scrub_tokens.py` scans a folder for leaked tokens.

1. Obtain COCO val2014 images and annotations from https://cocodataset.org (annotations: CC BY 4.0).
2. Run `pest_study.py` to build edits, train the conditions (frozen A, standard S, placebo-aware P, no-edit control N, DPO-style DS and DP) and write per-item prediction files.
3. Run `n10_hier_boot.py PRED_DIR OUT_DIR` for the hierarchical bootstrap (training seeds, then source images).
4. Run `make_figs2.py` to rebuild the figures from `results/`.

## Licences

- Code: MIT (see `LICENSE`).
- Our results, tables and figures: CC BY 4.0.
- MS COCO annotations: CC BY 4.0 (COCO Consortium). COCO images: Flickr Terms of Use.

## Citation

Please cite the paper and the Zenodo release (DOI to be added after the release).
