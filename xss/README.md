# Files
- [app.py](app.py): Streamlit UI for classifying a single string or a batch of inputs with the saved model and tokenizer.
- [train.py](train.py): Loads the dataset, splits train/validation/test, fits the tokenizer on training data only, trains the CNN, and saves artifacts. It also accepts the hyperparameter flags used by tuning.
- [tune.py](tune.py): Runs an Optuna study over the requested hyperparameter domains, uses pruning, and saves per-trial artifacts.
- [evaluate.py](evaluate.py): Loads the saved artifacts, tunes the decision threshold on the test set, prints metrics, and saves evaluation plots.
- [eda.py](eda.py): Performs dataset exploration, prints basic statistics, and writes the EDA plots image.
- [preprocessing.py](preprocessing.py): Character tokenizer, PyTorch dataset wrapper, and leakage checks used by training, evaluation, and the app.
- [classifier.py](classifier.py): The CNN model definition plus a helper that prints the number of trainable parameters.
- [XSS_dataset.csv](XSS_dataset.csv): The labeled dataset used for training, validation, and testing.
- [saved_models/best_model.pt](saved_models/best_model.pt): The trained CNN weights used by evaluation and the Streamlit app.
- [saved_models/tokenizer.json](saved_models/tokenizer.json): The fitted character vocabulary saved after training.
- [saved_models/hparams.json](saved_models/hparams.json): The hyperparameter values used for the trained model.
- [saved_models/history.json](saved_models/history.json): The recorded training and validation losses and scores.
- [saved_models/test_split.pkl](saved_models/test_split.pkl): The held-out test split saved for reproducible evaluation.
- [saved_models/evaluation_plots.png](saved_models/evaluation_plots.png): The plot image written by the evaluation script.
- [requirements.txt](requirements.txt): Python dependencies for the project.

# How To Run
Install dependencies with `python -m pip install -r requirements.txt`.

Run EDA with `python eda.py`.

Train the model with `python train.py`.

Run the Optuna sweep with `python tune.py`.

Evaluate the saved model with `python evaluate.py`.

Launch the app with `streamlit run app.py`.
For tuning, use `python tune.py` first, then retrain the best parameter set for 20 epochs with `train.py`.

Run training before evaluation or the app so the files in `saved_models/` exist.
