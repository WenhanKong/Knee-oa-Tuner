import argparse
import yaml
from .data_loader import get_data_generators
from .models import build_xception, build_resnet50, build_efficient_net_v2
from .tuners import get_bayesian_tuner, get_hyper_tuner
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model_name = config['model']['name']
    if model_name == 'xception':
        build_model = build_xception
    elif model_name == 'resnet50':
        build_model = build_resnet50
    elif model_name == 'efficient_net_v2':
        build_model = build_efficient_net_v2
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Initialize tuner
    tuner_name = config['tuner']['name']
    project_name = config.get('project_name', f"{model_name}_{tuner_name}")
    log_dir = f"logs/tuner"
    tb_log_dir = f"logs/tensorboard/{project_name}"
    model_dir = f'models/best_model_{project_name}.keras'
        
    if tuner_name == 'bayesian':
        tuner = get_bayesian_tuner(
            build_model,
            log_dir=log_dir,
            project_name=project_name,
            max_trials=config['tuner']['params']['max_trials']
        )
    elif tuner_name == 'hyperband':
        tuner = get_hyper_tuner(
            build_model,
            log_dir=log_dir,
            project_name=project_name,
            max_epochs=config['tuner']['params']['max_epochs']
        )
    else:
        raise ValueError(f"Unknown tuner: {tuner_name}")
    
    # Load data with configurable parameters
    train_gen, valid_gen = get_data_generators(
        batch_size=config['data']['batch_size'],
        image_size=config['data']['target_size']
    )
    tuner.search(
        train_gen,
        epochs=config['tuner']['params'].get('epochs', 30),
        validation_data=valid_gen,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            TensorBoard(log_dir=tb_log_dir)
        ],
        verbose=1
    )
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(model_dir)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    main(args.config)
