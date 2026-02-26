"""Shared job state store — single in-process job at a time."""

import asyncio
from typing import List
from enum import Enum


class JobStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobStore:
    def __init__(self):
        self.status: JobStatus = JobStatus.IDLE
        self.job_type: str = ""
        self.logs: List[str] = []
        self._subscribers: List[asyncio.Queue] = []

    def reset(self):
        self.status = JobStatus.RUNNING
        self.logs = []

    async def emit(self, line: str):
        self.logs.append(line)
        # Keep last 500 lines
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]
        for q in self._subscribers:
            await q.put(line)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]


# Singleton
job_store = JobStore()
