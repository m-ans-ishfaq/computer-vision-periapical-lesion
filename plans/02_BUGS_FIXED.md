# Bugs Found and Fixed

## Session 1 (Initial Code Review + Fixes)

### Bug 1: XML Labels Parsing — `int("PAI 3")` Crash (CRITICAL)
**File:** `src/data_utils.py:15`
**Issue:** `parse_labelme_xml()` stored the raw XML label text ("PAI 3") as `obj["label"]`. Downstream, `int(obj["label"])` would crash with `ValueError: invalid literal for int() with base 10: 'PAI 3'`. The dataset stores labels as strings, not integers.
**Fix:** Added `LABEL_MAP = {"PAI 3": 3, "PAI 4": 4, "PAI 5": 5}` with fallback for numeric strings. Labels are now stored as integers in the parsed annotation. Added `VALID_LABELS = {3, 4, 5}` and `None`-safe XML element access for malformed files.

### Bug 2: Only Classifier Head Trained, Backbone Random (CRITICAL)
**File:** `src/train_classification.py:47`
**Issue:** `optimizer = torch.optim.Adam(model.classifier.parameters(), ...)` only trained the final linear layers. The MobileNetV3 backbone stayed at random initialization (no pretrained weights passed to constructor). Result: ~33% accuracy (random).
**Fix:** Changed to `model.parameters()` so the full model gets trained.

### Bug 3: Segmentation Mask Label Remapping (CRITICAL)
**File:** `src/data_utils.py:234`
**Issue:** `prepare_segmentation_data()` used raw PAI IDs (3,4,5) as mask pixel values. UNet has 4 channels (0=bg, 1=PAI_3, 2=PAI_4, 3=PAI_5). CrossEntropyLoss requires values in [0, num_classes). Values 3-5 would silently produce wrong gradients.
**Fix:** Remap: `{3: 1, 4: 2, 5: 3}`. Skip unknown labels instead of mapping to background (0).

### Bug 4: Labels Not Moved to CPU Before numpy() (HIGH)
**File:** `src/train_classification.py:105`
**Issue:** `labels.numpy()` on line 105 vs `labels.cpu().numpy()` on line 74. Would crash if `DEVICE` is anything other than CPU (e.g., during testing on GPU).
**Fix:** Changed to `labels.cpu().numpy()`.

### Bug 5: Empty Manifest Crash (HIGH)
**File:** `src/data_utils.py:83`
**Issue:** `random.sample(manifest, sample_size)` crashes with `ValueError` if manifest is empty. Empty manifest can occur if `data/raw/` is empty or XML parsing finds zero annotations.
**Fix:** Added `if not manifest: return []` early guard.

### Bug 6: Bare `except:` Hides All Errors (HIGH)
**File:** `src/data_utils.py:65`
**Issue:** `except: pass` catches `KeyboardInterrupt`, `SystemExit`, and hides all XML parse failures with zero feedback. User sees empty manifest with no explanation.
**Fix:** Changed to `except Exception as e:` with `print(f"Warning: skipping {xml_path}: {e}")`.

### Bug 7: `CLASS_NAMES[label]` KeyError for Unexpected Labels (HIGH)
**File:** `src/data_utils.py:250`, `src/visualize.py:128-129`
**Issue:** `CLASS_NAMES[label]` crashes with `KeyError` if label is not 3/4/5 (e.g., from corrupted annotations). `get_image_level_label()` also crashed on non-integer labels.
**Fix:** Use `CLASS_NAMES.get(label, "Unknown")` and validate labels in `get_image_level_label()` against `VALID_LABELS`.

### Bug 8: Empty Validation Loader — `torch.cat([])` Crash (HIGH)
**File:** `src/train_segmentation.py:94-95`
**Issue:** If validation set is empty (e.g., tiny dataset with test split consuming all samples), `torch.cat([])` raises `RuntimeError`.
**Fix:** Added `if not all_preds: continue` guard in `train_segmentation()`. Same fix in `evaluate_segmentation()` returning `{"dice": 0.0, "iou": 0.0}`.

### Bug 9: Streamlit Progress Bar Loop (MEDIUM)
**File:** `app.py:119`
**Issue:** `for i, cls_name in [("PAI_3", 0), ...]` unpacked backwards. `i="PAI_3"`, `cls_name=0`. Worked by accident for progress value but variable naming misleading.
**Fix:** `for label_name, idx in ...` — and derive label list dynamically from `sorted(CLASS_NAMES.items())`.

### Bug 10: Hardcoded Labels in App (MEDIUM)
**File:** `app.py:119-124, 144`
**Issue:** `["PAI 3", "PAI 4", "PAI 5"]` hardcoded in 3 places. Would desync if `CLASS_NAMES` config changes.
**Fix:** Derive dynamically from `sorted(CLASS_NAMES.items())`.

### Bug 11: Temp File Collision in App (MEDIUM)
**File:** `app.py:132`
**Issue:** `_temp_upload.png` is shared between all uploads. Concurrent usage overwrites files + leaves stale temp files.
**Fix:** Use `tempfile.NamedTemporaryFile(delete=False)` with proper suffix from original filename.

### Bug 12: scan_dataset() Fragility (MEDIUM)
**File:** `src/data_utils.py:35`
**Issue:** Six fallback strategies with hardcoded paths. Multiple paths could crash on `os.path.exists(None)`.
**Fix:** Single `os.walk()` by keyword matching. Returns `None` for unfound dirs.

### Bug 13: Notebook Crash on Missing Data (MEDIUM)
**File:** `generate_notebooks.py:73`
**Issue:** After scan_dataset() returns None, `os.path.exists(None)` crashes.
**Fix:** Added `if original_dir is None: raise FileNotFoundError(...)` guard.

### Bug 14: Missing `bbox_inches="tight"` in Plot (LOW)
**File:** `src/visualize.py:30`
**Issue:** `plot_training_history` omitted `bbox_inches="tight"` unlike all other plot functions in the file.
**Fix:** Added `bbox_inches="tight"`.

### Bug 15: Inline Imports in visualize.py (LOW)
**File:** `src/visualize.py:105-106`
**Issue:** `import torch` and `from torchvision import transforms` inside function body. Hides missing dependencies until runtime.
**Fix:** Moved to module-level imports.

### Bug 16: Unused Requirements (LOW)
**File:** `requirements.txt` — removed 10 unused packages
**Removed:** `torchaudio`, `scikit-image`, `scipy`, `seaborn`, `albumentations`, `pycocotools`, `lxml`, `xmltodict`, `patool`, `rarfile`
**Why:** Some (pycocotools, patool) require C compilers on Windows, causing install failures for no benefit.

## Session 1 Summary
| Severity | Count | Labs |
|----------|-------|------|
| CRITICAL | 3 | XML label parsing, optimizer scope, seg mask remap |
| HIGH     | 5 | `labels.cpu()`, empty manifest, bare except, KeyError, empty cat |
| MEDIUM   | 5 | Progress bar, hardcoded labels, temp file, scan_dataset, notebook guard |
| LOW      | 3 | bbox_inches, inline imports, unused deps |
