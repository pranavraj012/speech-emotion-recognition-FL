# Federated SER Presentation Content

## Slide 1: Title
- Privacy-Preserving Speech Emotion Recognition
- Federated Learning Simulation on RAVDESS
- Your name, date, course/project context

Speaker notes:
- This project builds a speech emotion recognition system without moving raw user audio to a central server.
- The implementation uses a Flower federated setup with local client simulation.

## Slide 2: Problem Statement
- Goal: classify emotions from speech while preserving privacy
- Classes: neutral, happy, sad, angry, fearful, disgusted
- Target: at least 85% test accuracy

Speaker notes:
- In centralized ML, all user data must be uploaded.
- In FL, users keep data local and only share model updates.

## Slide 3: Data Identified and Prepared
- Dataset: RAVDESS speech audio clips
- Filtered to 6 classes from the original 8
- Audio transformed into 1024-dim HuBERT embeddings
- StandardScaler fitted on train split

Speaker notes:
- HuBERT gives richer speech features than classic MFCC-only approaches.
- Each sample becomes a fixed vector, making FL model training efficient.

## Slide 4: Train/Test Split and FL Data Flow
- Global split done before FL partitioning
- Current run: 80/20 train/test
- Only train split is partitioned across clients
- Test split is never used by clients

Speaker notes:
- This is the correct way to evaluate FL fairly.
- Clients train on disjoint training shards and server evaluates on holdout test set.

## Slide 5: FL Environment Implemented
- Framework: Flower
- Strategy: FedAvg
- Simulation: 5 local clients + 1 server on one machine
- Communication: Flower gRPC internals
- Device: CUDA-enabled local training when available

Speaker notes:
- We simulate distributed training locally for reproducibility.
- Architecture can be extended to real multi-device deployment.

## Slide 6: Model and Training Setup
- Model: MLP
- Architecture: 1024 -> 256 -> 6
- Optimizer: Adam
- Local training: mini-batch, multiple local epochs
- DP noise support exists but disabled in this accuracy run

Speaker notes:
- MLP improves representation power over a single linear classifier.
- Privacy-accuracy tradeoff can be tuned by epsilon.

## Slide 7: Tech Stack
- Python, PyTorch, TorchAudio
- Hugging Face Transformers (HuBERT)
- Flower
- NumPy, scikit-learn, Joblib
- Streamlit + streamlit-audiorecorder

Speaker notes:
- The stack is lightweight and reproducible with pip in a venv.

## Slide 8: Results
- Final test accuracy: around 93%
- Best accuracy exceeds 85% target
- Stable high accuracy after initial rounds

Visual:
- Use presentation_assets/accuracy_curve.png

Speaker notes:
- The model converges quickly and stays above threshold after early rounds.

## Slide 9: Visual Evidence
- Global test accuracy curve
- Global test loss curve
- Confusion matrix
- Per-class accuracy chart

Visuals:
- presentation_assets/accuracy_curve.png
- presentation_assets/loss_curve.png
- presentation_assets/confusion_matrix.png
- presentation_assets/class_accuracy.png

Speaker notes:
- These visuals show both overall and class-specific behavior.

## Slide 10: What I Learned
- FL requires strict split discipline (holdout first, then client partition)
- Communication orchestration is as important as model quality
- Privacy mechanisms affect utility; tuning is essential
- Small architecture changes can dramatically improve convergence

Speaker notes:
- The final MLP setup fixed low-accuracy behavior from the older linear setup.

## Slide 11: Demo
- Streamlit app to test your own voice
- Upload custom audio file
- Record live voice in browser
- Real-time emotion prediction with confidence bars

Run demo:
- streamlit run streamlit_app.py

Speaker notes:
- This gives an end-user friendly view of the trained FL model.

## Slide 12: Future Work
- Add true multi-machine FL clients
- Add stronger privacy with calibrated DP while preserving performance
- Add robust augmentation for noisy real-world audio
- Add richer production monitoring and model cards
