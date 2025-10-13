"""Custom callbacks for training monitoring and evaluation."""

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from pathlib import Path


class ProgressCallback(BaseCallback):
    """
    Custom callback for logging training progress.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """
        Called at each step.
        
        Returns:
            True to continue training
        """
        # Log episode statistics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
            
            if self.verbose > 0:
                print(f"Episode reward: {ep_info['r']:.2f}, length: {ep_info['l']}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if len(self.episode_rewards) > 0:
            print(f"\nTraining completed!")
            print(f"Mean episode reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Mean episode length: {np.mean(self.episode_lengths):.2f}")


class SaveBestModelCallback(BaseCallback):
    """
    Callback for saving the best model based on evaluation reward.
    """
    
    def __init__(self, eval_env, save_path: str, eval_freq: int = 10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = Path(save_path)
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        """
        Evaluate model and save if it's the best so far.
        
        Returns:
            True to continue training
        """
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=10,
                deterministic=True
            )
            
            if self.verbose > 0:
                print(f"Eval at step {self.num_timesteps}: mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Save if best
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                save_file = self.save_path / "best_model"
                self.model.save(save_file)
                
                if self.verbose > 0:
                    print(f"New best model saved with reward: {mean_reward:.2f}")
        
        return True
