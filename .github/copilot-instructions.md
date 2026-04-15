Purpose
This file gives Copilot sessions repository-specific guidance: how to run, test, and reason about the codebase so suggestions are accurate and actionable.

Quick commands (what exists)
- Install runtime deps (recommended):
  pip install click numpy scikit-learn scipy librosa pyyaml soundfile joblib pytest pytest-cov

- Run CLI locally (module entrypoint):
  python -m src
  # If package installed via setup/entry points: underwater-audio --help

- Run full test suite:
  pytest tests/ -v

- Run with coverage:
  pytest tests/ --cov=src --cov-report=term

- Run a single test file (example):
  pytest tests/test_audio_loader.py -v

- Run a single test class or test function (example):
  pytest tests/test_classifier.py::TestSoundClassifierTraining -v
  pytest tests/test_feature_extractor.py::test_nmf_output -v

- Lint/build scripts: none detected in repo. Do not assume repo-supplied linters or builders; add CI steps if needed.

High-level architecture (big picture)
- Purpose: command-line tool for underwater-vehicle sound classification with an end-to-end pipeline (audio -> mel spectrogram -> NMF -> feature vector -> classifier).
- Top-level components:
  - src/cli.py: Click-based command group (train, classify, batch-classify, analyze, config).
  - src/pipeline.py: UnderWaterAudioPipeline — orchestrates preprocessor, feature extractor, and classifier.
  - src/audio_loader.py: audio I/O (librosa-based loaders, mono/stereo handling).
  - src/preprocessor.py: mel-spectrogram, STFT, spectral gating, HPSS.
  - src/feature_extractor.py: NMFFeatureExtractor (sklearn or custom NMF) and feature-vector assembly.
  - src/classifier.py: SoundClassifier — trains RF/SVM/GB, predict/probabilities, save/load via joblib.
  - config.yaml: canonical pipeline defaults (preprocessor, nmf, classifier, label_names).
- Runtime flow (summary): validate data dir -> load audio -> preprocess to mel-specs -> run NMF -> extract features -> train/evaluate classifier -> save model + config. For inference: load model, auto-detect config alongside model, preprocess audio, extract features, predict.

Key conventions and repo-specific patterns
- Config auto-detection: when classifying, the CLI attempts to find config.yaml in the model's directory. Prefer saving config next to models (models/config.yaml).
- Label mapping: class names are mapped to integer IDs by alphabetical sort; IDs must be contiguous starting at 0. Tests and code rely on this mapping.
- NMF & feature sizing: config.nmf.n_components controls NMF basis count. Feature vector length = n_components * 8 (4 moments from W, 4 from H) — used across pipeline and tests.
- Model serialization: use SoundClassifier.save/load (joblib). Do not hand-edit or manually pickle model internals.
- CLI pattern: Click group with subcommands; subcommands expose --verbose flag to enable DEBUG-like output. Exit codes are standardized (0 success, 1 runtime/processing error, 2 usage error).
- Tests: pytest + Click CliRunner. Tests use synthetic audio fixtures and set random seed 42 for reproducibility. Tests are organized per module (audio_loader, preprocessor, feature_extractor, classifier, pipeline, cli).
- Data layout expectations: training_data/<ClassName>/*.wav (one directory per class). Code validates presence of subdirectories and audio files.
- Error handling: commands catch exceptions, print colored messages and exit with non-zero codes; batch-classify handles per-file errors and continues.

Files to consult when composing suggestions
- src/cli.py (entry and command wiring)
- src/pipeline.py, src/classifier.py, src/feature_extractor.py, src/preprocessor.py, src/audio_loader.py
- config.yaml (defaults & parameter names)
- tests/conftest.py and tests/* for expected behaviors tested by CI

Notes for Copilot responses (how to be most helpful)
- When suggesting code changes, reference the pipeline pieces above and keep changes localized to the relevant module (e.g., feature changes in feature_extractor.py, CLI args in cli.py, serialization in classifier.py).
- When recommending config keys, use the existing config.yaml keys/names and types (preprocessor.n_fft, nmf.n_components, classifier.model_type, label_names mapping).
- When producing CLI examples, prefer using "python -m src" (reproducible for local dev) and include --verbose for debug output.
- For tests, suggest running specific pytest invocations shown above; prefer adding tests under tests/ and reusing fixtures from tests/conftest.py.

Repository-specific gotchas for generated code
- Do not change the label mapping logic (alphabetical -> contiguous ints) unless accompanied by updates to tests and CLI helpers that rely on it.
- Keep model serialization compatible with joblib-based SoundClassifier.save/load.
- Maintain config auto-detection behavior if adding alternative storage locations.

Integration notes
- No repository AI assistant configs detected (CLAUDE.md, AGENTS.md, .windsurfrules, .cursor rules, AIDER_CONVENTIONS, etc.).

Summary
Created guidance that lists actionable commands, the big-picture architecture, and the key conventions Copilot should follow when editing or suggesting changes in this repository.

Would you like any additions (for example: CI snippets, example model files, or explicit recommended linting tools to standardize)?
