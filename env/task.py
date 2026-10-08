"""
Task Module

This module defines the Task class representing computational tasks
that arrive in the cloud environment and need to be scheduled to servers.

Each task has:
- CPU requirement: The amount of CPU resources needed
- Processing time: How long the task takes to complete
- Arrival time: When the task entered the system
- Completion time: When the task finished (set after completion)
"""

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class Task:
    """
    Represents a computational task in the cloud environment.
    
    A task is the fundamental unit of work that needs to be scheduled.
    Tasks arrive dynamically and must be assigned to servers for processing.
    
    Attributes:
        task_id: Unique identifier for the task
        cpu_requirement: CPU resources needed (e.g., 10-50 units)
        processing_time: Number of time steps to complete the task
        arrival_time: Time step when the task arrived
        start_time: Time step when processing began (None if not started)
        completion_time: Time step when task completed (None if not done)
        assigned_server: ID of server this task is assigned to (None if unassigned)
    
    Example:
        task = Task(
            task_id=1,
            cpu_requirement=25.0,
            processing_time=5,
            arrival_time=10
        )
    """
    
    task_id: int
    cpu_requirement: float
    processing_time: int
    arrival_time: int
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    assigned_server: Optional[int] = None
    
    # Track remaining processing time (initialized to processing_time)
    remaining_time: int = field(init=False)
    
    def __post_init__(self):
        """Initialize remaining time after dataclass initialization."""
        self.remaining_time = self.processing_time
    
    @property
    def is_completed(self) -> bool:
        """Check if the task has been completed."""
        return self.completion_time is not None
    
    @property
    def is_started(self) -> bool:
        """Check if the task has started processing."""
        return self.start_time is not None
    
    @property
    def is_assigned(self) -> bool:
        """Check if the task has been assigned to a server."""
        return self.assigned_server is not None
    
    def get_waiting_time(self, current_time: int) -> int:
        """
        Calculate the waiting time for this task.
        
        Waiting time is the duration from arrival to start of processing.
        If the task hasn't started yet, returns time waited so far.
        
        Args:
            current_time: Current simulation time step
            
        Returns:
            Number of time steps the task has waited
        """
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return current_time - self.arrival_time
    
    def get_latency(self) -> Optional[int]:
        """
        Calculate the total latency (turnaround time) for this task.
        
        Latency = Waiting Time + Processing Time
                = Completion Time - Arrival Time
        
        Returns:
            Total latency if task is completed, None otherwise
        """
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None
    
    def start(self, current_time: int) -> None:
        """
        Mark the task as started.
        
        Args:
            current_time: Current simulation time step
        """
        if self.start_time is None:
            self.start_time = current_time
    
    def process(self, current_time: int) -> bool:
        """
        Process the task for one time step.
        
        Decrements remaining time and marks completion if done.
        
        Args:
            current_time: Current simulation time step
            
        Returns:
            True if task completed this step, False otherwise
        """
        # Start the task if not already started
        if not self.is_started:
            self.start(current_time)
        
        # Decrement remaining time
        self.remaining_time -= 1
        
        # Check if completed
        if self.remaining_time <= 0:
            self.completion_time = current_time
            return True
        
        return False
    
    def assign_to_server(self, server_id: int) -> None:
        """
        Assign this task to a specific server.
        
        Args:
            server_id: ID of the server to assign to
        """
        self.assigned_server = server_id
    
    def __repr__(self) -> str:
        """String representation of the task."""
        status = "completed" if self.is_completed else \
                 "processing" if self.is_started else \
                 "assigned" if self.is_assigned else "pending"
        return (f"Task(id={self.task_id}, cpu={self.cpu_requirement:.1f}, "
                f"time={self.processing_time}, status={status})")


class TaskGenerator:
    """
    Factory class for generating random tasks.
    
    This class encapsulates the logic for creating tasks with
    randomized parameters within configured ranges.
    
    Attributes:
        cpu_range: Tuple of (min, max) CPU requirement
        time_range: Tuple of (min, max) processing time
        arrival_prob: Probability of task arrival each step
        task_counter: Counter for generating unique task IDs
    
    Example:
        generator = TaskGenerator(
            cpu_range=(10.0, 50.0),
            time_range=(1, 10),
            arrival_prob=0.7
        )
        task = generator.generate(current_time=5)
    """
    
    def __init__(
        self,
        cpu_range: tuple = (10.0, 50.0),
        time_range: tuple = (1, 10),
        arrival_prob: float = 0.7,
        seed: Optional[int] = None
    ):
        """
        Initialize the task generator.
        
        Args:
            cpu_range: (min, max) CPU requirement for tasks
            time_range: (min, max) processing time for tasks
            arrival_prob: Probability of generating a task
            seed: Random seed for reproducibility
        """
        self.cpu_range = cpu_range
        self.time_range = time_range
        self.arrival_prob = arrival_prob
        self.task_counter = 0
        
        if seed is not None:
            random.seed(seed)
    
    def should_generate(self) -> bool:
        """
        Determine if a task should be generated this time step.
        
        Returns:
            True if a task should be generated based on arrival probability
        """
        return random.random() < self.arrival_prob
    
    def generate(self, current_time: int) -> Task:
        """
        Generate a new task with random parameters.
        
        Args:
            current_time: Current simulation time step (becomes arrival time)
            
        Returns:
            A new Task instance with randomized parameters
        """
        self.task_counter += 1
        
        # Generate random CPU requirement
        cpu_req = random.uniform(self.cpu_range[0], self.cpu_range[1])
        
        # Generate random processing time
        proc_time = random.randint(self.time_range[0], self.time_range[1])
        
        return Task(
            task_id=self.task_counter,
            cpu_requirement=cpu_req,
            processing_time=proc_time,
            arrival_time=current_time
        )
    
    def maybe_generate(self, current_time: int) -> Optional[Task]:
        """
        Potentially generate a task based on arrival probability.
        
        Args:
            current_time: Current simulation time step
            
        Returns:
            A new Task if generated, None otherwise
        """
        if self.should_generate():
            return self.generate(current_time)
        return None
    
    def reset(self) -> None:
        """Reset the task counter for a new episode."""
        self.task_counter = 0
