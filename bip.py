import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# # Create the Car Racing environment
env = gym.make('BipedalWalker-v3')
# env = gym.make('BipedalWalker-v3',render_mode="human")

# # Wrap the environment with DummyVecEnv
env = DummyVecEnv([lambda: env])

# env = Monitor(env, "./logs")

# # Define the PPO model
model = SAC('MlpPolicy', env, verbose=0, tensorboard_log="./models/SAC/logs/")


# # Create an evaluation callback to evaluate the model during training
eval_callback = EvalCallback(env, best_model_save_path='./models/SAC/models/',
                            log_path='./models/SAC/logs/', eval_freq=1000,
                            deterministic=True, render=False)

# # Create a checkpoint callback to save the model at regular intervals
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/SAC/models/')

# # Train the model
model.learn(total_timesteps=300000, callback=[eval_callback, checkpoint_callback], progress_bar=True)

# # Evaluate the trained model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

# # Print the mean reward achieved during evaluation
print("Mean reward:", mean_reward)

# # Save the final trained model
model.save("./models/SAC/models/final_model")

# # # Close the environment
del model 

# # VISUALIZAR O MODELO TREINADO
# model = SAC.load("./models/SAC/models/final_model", env=env)

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
# print (f"recompensa media #{mean_reward} - desvio padrÃ£o +/- #{std_reward}")

# Enjoy trained agent
# vec_env = model.get_env()
# obs = vec_env.reset()

# try:
#     while True:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = vec_env.step(action)
         
#         if done:
#             obs = vec_env.reset()
# except KeyboardInterrupt:
#     pass