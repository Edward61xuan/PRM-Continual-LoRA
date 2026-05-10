"""
Class-Incremental Manager for continual learning.

Manages the task sequence: splits a full class list into num_tasks groups,
optionally shuffles them, and provides an iterator over (task_id, task_classes).
"""

import numpy as np
from copy import deepcopy
from typing import Literal


class ClassIncrementalManager:
    """Manages class-incremental task scheduling.

    Splits the full class list into `num_tasks` groups of equal size,
    with optional class-level or task-level shuffling controlled by a seed.

    Usage:
        manager = ClassIncrementalManager(class_list, num_tasks=10, seed=2024)
        for task_id, task_classes in manager:
            # train on task_classes, evaluate on all seen classes so far
            ...

    Args:
        class_list:    Full list of class indices.
        num_tasks:     Number of incremental tasks.
        rand_seed:     Random seed for reproducible class shuffling.
        shuffle:       Whether to shuffle classes across tasks.
        shuffle_level: 'class' shuffles individual classes; 'task' shuffles task order.
    """

    def __init__(
        self,
        class_list,
        num_tasks,
        rand_seed=0,
        shuffle=True,
        shuffle_level: Literal["class", "task"] = "class",
    ):
        assert (
            len(class_list) % num_tasks == 0
        ), f"Classes ({len(class_list)}) must be divisible by num_tasks ({num_tasks})"

        rng = np.random.Generator(np.random.PCG64(rand_seed))
        task_class_array = np.array(class_list, dtype=np.int64)

        if shuffle:
            if shuffle_level == "class":
                rng.shuffle(task_class_array)
                task_class_array = np.reshape(task_class_array, [num_tasks, -1])
            elif shuffle_level == "task":
                task_class_array = np.reshape(task_class_array, [num_tasks, -1])
                rng.shuffle(task_class_array)
        else:
            task_class_array = np.reshape(task_class_array, [num_tasks, -1])

        self._all_classes = task_class_array.flatten().tolist()
        self._task_class_list = task_class_array.tolist()
        self._current_taskid = -1
        self._num_tasks = num_tasks

    @property
    def current_taskid(self):
        assert self._current_taskid >= 0, "Not initialized (call next() first)"
        return self._current_taskid

    @property
    def all_classes(self):
        return self._all_classes

    @property
    def task_class_list(self):
        return self._task_class_list

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def num_classes_per_task(self):
        return len(self._task_class_list[0])

    @property
    def current_task_classes(self):
        return deepcopy(self._task_class_list[self.current_taskid])

    @property
    def sofar_task_classes(self):
        """All classes seen so far (tasks 0..current), flattened."""
        classes = []
        for i in range(self.current_taskid + 1):
            classes.extend(self._task_class_list[i])
        return deepcopy(classes)

    def get_classes(self, taskid):
        """Get the class list for a specific task."""
        return self._task_class_list[taskid]

    def __iter__(self):
        return self

    def __next__(self):
        self._current_taskid += 1
        if self._current_taskid >= len(self):
            self._current_taskid = -1
            raise StopIteration()
        return self.current_taskid, self.current_task_classes

    def __len__(self):
        return self.num_tasks
