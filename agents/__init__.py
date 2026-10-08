"""
Agents Module

This module contains the RL agents:
- DQNAgent: Deep Q-Network agent for learning optimal scheduling
- PrioritizedReplayBuffer: Experience replay buffer for stable training
"""

from utils.replay_buffer import PrioritizedReplayBuffer
from agents.dqn_agent import DQNAgent, DuelingDQN

# Backward compatibility for older imports.
ReplayBuffer = PrioritizedReplayBuffer

__all__ = ['DQNAgent', 'DuelingDQN', 'PrioritizedReplayBuffer', 'ReplayBuffer']
