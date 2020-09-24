## Run

Place expert dataset in this directory
```bash
python main.py --env-name "Circles-v0" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay --use-proper-time-limits --gail
```

The file will contain a dictionary of tensors including states and actions. 
Once loaded, `states.shape` is `[num_traj, traj_len, state_dim] = [500, 1000, 10]`. Similarly, `actions.shape` is `[num_traj, traj_len, action_dim]  = [500, 1000, 2]`.
