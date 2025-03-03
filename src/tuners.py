import keras_tuner as kt

def get_bayesian_tuner(build_model, project_name, log_dir, max_trials=30):
    return kt.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory=log_dir,
        project_name=project_name,
        overwrite=False
    )


def get_hyper_tuner(build_model, project_name, log_dir, max_epochs=100):
    return kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=1,
        directory=log_dir,
        project_name=project_name,
        overwrite=False
    )