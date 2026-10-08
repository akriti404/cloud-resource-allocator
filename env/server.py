from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env.task import Task


@dataclass
class Server:
    server_id: int
    cpu_capacity: float = 100.0
    max_queue_length: int = 20
    
    task_queue: deque = field(default_factory=deque, repr=False)
    current_tasks: List = field(default_factory=list, repr=False)
    current_cpu_usage: float = field(default=0.0, repr=False)
    total_tasks_processed: int = field(default=0, repr=False)
    total_latency: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate initialization values."""
        if self.cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than 0")
    
    @property
    def queue_length(self) -> int:
        """Get the current number of tasks in the queue."""
        return len(self.task_queue)
    
    @property
    def num_active_tasks(self) -> int:
        """Get the number of tasks currently being processed."""
        return len(self.current_tasks)
    
    @property
    def cpu_utilization(self) -> float:
        return min(1.0, self.current_cpu_usage / self.cpu_capacity)
    
    @property
    def is_overloaded(self) -> bool:
        return (self.cpu_utilization > 0.9 or 
                self.queue_length >= self.max_queue_length)
    
    @property
    def available_cpu(self) -> float:
        return max(0.0, self.cpu_capacity - self.current_cpu_usage)
    
    @property
    def can_accept_task(self) -> bool:
        return self.queue_length < self.max_queue_length
    
    def add_task(self, task: 'Task') -> bool:
        if not self.can_accept_task:
            return False
        
        task.assign_to_server(self.server_id)
        self.task_queue.append(task)
        return True
    
    def _start_next_task(self, current_time: int) -> bool:
        if len(self.task_queue) == 0:
            return False
        
        # Peek at the next task
        next_task = self.task_queue[0]
        
        # Check if we have enough CPU capacity
        if self.current_cpu_usage + next_task.cpu_requirement <= self.cpu_capacity:
            # Remove from queue and start processing
            task = self.task_queue.popleft()
            task.start(current_time)
            self.current_tasks.append(task)
            self.current_cpu_usage += task.cpu_requirement
            return True
        
        return False
    
    def step(self, current_time: int) -> List['Task']:
        completed_tasks = []
        tasks_to_remove = []
        
        for task in self.current_tasks:
            if task.process(current_time):
                completed_tasks.append(task)
                tasks_to_remove.append(task)
                self.total_tasks_processed += 1
                
                latency = task.get_latency()
                if latency is not None:
                    self.total_latency += latency
        
        for task in tasks_to_remove:
            self.current_tasks.remove(task)
            self.current_cpu_usage -= task.cpu_requirement
        
        self.current_cpu_usage = max(0.0, self.current_cpu_usage)
        
        while self._start_next_task(current_time):
            pass
        
        return completed_tasks
    
    def get_estimated_wait_time(self, task: 'Task') -> int:
        active_remaining = sum(t.remaining_time for t in self.current_tasks)
        
        queued_time = sum(t.processing_time for t in self.task_queue)
        
        return active_remaining + queued_time
    
    def get_state(self) -> Tuple[float, int]:
        return (self.cpu_utilization, self.queue_length)
    
    def get_average_latency(self) -> float:
        if self.total_tasks_processed == 0:
            return 0.0
        return self.total_latency / self.total_tasks_processed
    
    def reset(self) -> None:
        self.task_queue.clear()
        self.current_tasks.clear()
        self.current_cpu_usage = 0.0
        self.total_tasks_processed = 0
        self.total_latency = 0
    
    def get_info(self) -> dict:
        return {
            'server_id': self.server_id,
            'cpu_utilization': self.cpu_utilization,
            'queue_length': self.queue_length,
            'active_tasks': self.num_active_tasks,
            'is_overloaded': self.is_overloaded,
            'total_processed': self.total_tasks_processed,
            'avg_latency': self.get_average_latency()
        }
    
    def __repr__(self) -> str:
        return (f"Server(id={self.server_id}, "
                f"cpu={self.cpu_utilization:.1%}, "
                f"queue={self.queue_length})")
