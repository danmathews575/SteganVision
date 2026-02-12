# Repository Cleanup and Reorganization Plan

## 1. Clean Folder Tree

```ascii
MajorP/
├── .gitignore                   # New research-grade gitignore
├── README.md                    # New Global README (rename README_ROOT.md)
├── audio_steganography/         # [Future] Audio Project
│   └── .gitkeep
├── backend/                     # [Future] Backend API
│   └── .gitkeep
├── docs/                        # Global Documentation
│   ├── DATASET.md
│   └── ...
├── experiments/                 # Shared Experiments & Logs
│   ├── logs/
│   └── ...
├── frontend/                    # [Future] Frontend UI
│   └── .gitkeep
├── image_steganography/         # [Moved] Image Project (Frozen)
│   ├── checkpoints/             # (Best models only)
│   ├── configs/
│   ├── data/                    # (CelebA, etc.)
│   ├── notebooks/
│   ├── outputs/
│   ├── results/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── PROJECT_STRUCTURE.md
│   ├── QUICKSTART.md
│   ├── README.md                # (Original README)
│   ├── requirements.txt
│   ├── setup.py
│   └── test_forward.py
└── text_steganography/          # [Future] Text Project
    └── .gitkeep
```

## 2. Cleanup Checklist

- [x] Create new module directories (`image_`, `text_`, `audio_`, etc.)
- [x] Create `.gitignore` (Done)
- [x] Create new Root `README.md` (Drafted as `README_ROOT.md`)
- [ ] **ACTION REQUIRED**: Move Image Steganography files to `image_steganography/`
- [ ] **ACTION REQUIRED**: Rename `README_ROOT.md` to `README.md`
- [ ] **ACTION REQUIRED**: Verify imports/paths (Scripts should run from `image_steganography/` root or adjust PYTHONPATH)

## 3. Git-Safe Commands & Cleanup Plan

### Step 1: Commit any pending changes (Safety First)
```bash
git add .
git commit -m "Pre-cleanup state"
```

### Step 2: Remove Junk (Safe)
```bash
# Remove execution of this if you are not sure, relying on .gitignore is safer.
# git rm -r --cached __pycache__
# git rm -r --cached .vscode
```

### Step 3: Move Image Project Files
Run these commands in your Git Bash / Terminal:

```bash
# Move source code and config
git mv src image_steganography/
git mv scripts image_steganography/
git mv configs image_steganography/
git mv notebooks image_steganography/
git mv tests image_steganography/

# Move data and artifacts
# Note: 'data' might be heavy. If it's ignored, git mv might warn.
# If 'data' is NOT tracked, just use 'mv'.
git mv data image_steganography/ || mv data image_steganography/
git mv checkpoints image_steganography/
git mv outputs image_steganography/
git mv results image_steganography/

# Move root files
git mv setup.py image_steganography/
git mv requirements.txt image_steganography/
git mv test_forward.py image_steganography/

# Move Documentation
git mv PROJECT_STRUCTURE.md image_steganography/
git mv QUICKSTART.md image_steganography/
git mv README.md image_steganography/README_original.md
```

### Step 4: Finalize
```bash
# Rename new README
mv README_ROOT.md README.md
git add README.md .gitignore image_steganography text_steganography audio_steganography backend frontend experiments

# Commit
git commit -m "Refactor: Restructure repository for multimodal steganography"
```

### 4. Implementation Notes regarding `setup.py`
The `setup.py` defines `package_dir={"": "src"}`. moving it to `image_steganography/` along with `src/` maintains this relative relationship.
To work on the image project, you should now navigate to `image_steganography/` and run `pip install -e .` or run scripts from there.
