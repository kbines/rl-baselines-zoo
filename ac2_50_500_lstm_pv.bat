python train.py^
 --algo a2c^
 --env PortfolioAllocation-v0^
 -tb tensorboard^
 --log-folder ac2_50_500_lstm_pv/log^
 --n-timesteps 2517^
 --save-freq 10^
 --n-trials 20^
 --optimize^
 --verbose 1^
 --env-kwargs sample_size:500 reward_function:"'portfolio_value'"