import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler
from stable_baselines import SAC, TD3
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import VecNormalize, VecEnv
from utils import linear_schedule
from tensorflow import nn as nn

# Load mpi4py-dependent algorithms only if mpi is installed. (c.f. stable-baselines v2.10.0)
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines import DDPG
else:
    DDPG = None
del mpi4py

from .callbacks import TrialEvalCallback


def hyperparam_optimization(algo, model_fn, env_fn, n_trials=10, n_timesteps=5000, hyperparams=None,
                            n_jobs=1, sampler_method='random', pruner_method='halving',
                            seed=0, verbose=1, storage=None, study_name=None):
    """
    :param algo: (str)
    :param model_fn: (func) function that is used to instantiate the model
    :param env_fn: (func) function that is used to instantiate the env
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param hyperparams: (dict)
    :param n_jobs: (int) number of parallel jobs
    :param sampler_method: (str)
    :param pruner_method: (str)
    :param seed: (int)
    :param verbose: (int)
    :return: (pd.Dataframe) detailed result of the optimization
    """
    # TODO: eval each hyperparams several times to account for noisy evaluation
    # TODO: take into account the normalization (also for the test env -> sync obs_rms)
    if hyperparams is None:
        hyperparams = {}

    n_startup_trials = 10
    # test during 5 episodes
    n_eval_episodes = 5
    # evaluate every 20th of the maximum budget per iteration
    n_evaluations = 20
    eval_freq = int(n_timesteps / n_evaluations)

    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler_method == 'tpe':
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == 'skopt':
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler_method))

    if pruner_method == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == 'median':
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == 'none':
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner_method))

    if verbose > 0:
        print("Sampler: {} - Pruner: {}".format(sampler_method, pruner_method))

    #study = optuna.create_study(sampler=sampler, pruner=pruner)
    study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
            direction="maximize",
    )
    algo_sampler = HYPERPARAMS_SAMPLER[algo]

    def objective(trial):

        kwargs = hyperparams.copy()

        trial.model_class = None
        if algo == 'her':
            trial.model_class = hyperparams['model_class']

        # Hack to use DDPG/TD3 noise sampler
        if algo in ['ddpg', 'td3'] or trial.model_class in ['ddpg', 'td3']:
            trial.n_actions = env_fn(n_envs=1).action_space.shape[0]
        kwargs.update(algo_sampler(trial))

        model = model_fn(**kwargs)

        eval_env = env_fn(n_envs=1, eval_env=True)
        # Account for parallel envs
        eval_freq_ = eval_freq
        if isinstance(model.get_env(), VecEnv):
            eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)
        # TODO: use non-deterministic eval for Atari?
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=n_eval_episodes,
                                          eval_freq=eval_freq_, deterministic=True)

        try:
            model.learn(n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        cost = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return cost

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


def sample_a2c_params(trial):
    """
    Sampler for A2C hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.9999])
    n_steps = trial.suggest_categorical('n_steps', [1, 5, 10, 30, 90])
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear','constant'])
    learning_rate = trial.suggest_categorical('lr', [0.0001, 0.0007, 0.001, 0.01])
    ent_coef = trial.suggest_categorical('ent_coef', [0.00000001, 0.000001, 0.0001, 0.1])
    vf_coef = trial.suggest_categorical('vf_coef', [0, 0.25, 0.5, 0.75, 1])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.7, 1, 2, 5])
    n_lstm=trial.suggest_categorical('n_lstm', [64,128,256])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'leaky_relu'])
    net_arch = trial.suggest_categorical('net_arch', ["tiny", "small", "medium", "small3", "medium3"])
    #360k variations
    net_arch = {
        "tiny": [8, 'lstm', dict(pi=[32, 32], vf=[32, 32])],
        "small": [64, 'lstm', dict(pi=[256, 256], vf=[256, 256])],
        "medium": [128, 'lstm', dict(pi=[256, 256], vf=[256, 256])],
        "small3": [64, 'lstm', dict(pi=[64, 64, 64], vf=[64, 64, 64])],
        "medium3": [128, 'lstm', dict(pi=[256, 256, 256], vf=[256, 256, 256])],
    }[net_arch]
    
    activation_fn = {"tanh": nn.tanh, "relu": nn.relu, "leaky_relu": nn.leaky_relu}[activation_fn]

    return {
        'gamma': gamma,
        'n_steps': n_steps,
        'lr_schedule': lr_schedule,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm' : max_grad_norm,
        'policy_kwargs': dict(
            n_lstm=n_lstm,
            net_arch=net_arch,
            act_fun=activation_fn
        )
    }

HYPERPARAMS_SAMPLER = {
    'a2c': sample_a2c_params
}
