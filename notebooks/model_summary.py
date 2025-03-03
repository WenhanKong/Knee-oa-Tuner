# %%
import sys
import os

# Get the project root directory (modify if necessary)
project_root = os.path.abspath("..")  # Adjust based on where your notebook is located

# Add to Python's sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
    
from tensorflow import keras

# model_path = '/WAVE/users2/unix/wkong/ml240/models/best_model_xception_hyperband.keras'
# # Load the saved model
# best_model = keras.models.load_model(model_path)

# %%
from src.tuners import get_bayesian_tuner, get_hyper_tuner
from src.models import build_xception

tuner = get_bayesian_tuner(
    build_xception,
    log_dir='/WAVE/users2/unix/wkong/ml240/logs/tuner/xception_bayesian',
    project_name='xception_bayesian'
)

best_hps = tuner.get_best_hyperparameters()[0]

# Print best hyperparameters
print("Best Hyperparameters:")
for hp_name in best_hps.values.keys():
    print(f"{hp_name}: {best_hps.get(hp_name)}")

# %%
import json

with open("best_hps.json", "w") as f:
    json.dump(best_hps.values, f)

# %%
