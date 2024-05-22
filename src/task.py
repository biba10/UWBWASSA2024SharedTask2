from enum import Enum


class Task(Enum):
    TASK1 = "task1"
    TASK2 = "task2"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)