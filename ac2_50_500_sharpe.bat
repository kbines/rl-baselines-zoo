python train.py^
 --algo a2c --env PortfolioAllocation-v0^
 --n-timesteps 20^
 --optimization-log-path ac2_50_500_sharpe/opt_log^
 --log-folder ac2_50_500_sharpe/log^
 --save-freq 10^
 --vec-env dummy^
 --n-trials 5^
 --optimize-hyperparameters^
 --study-name ac2_50_500_sharpe^
 --verbose 1^
 --env-kwargs sample_size:500