# Listener.py
from abc import ABC, abstractmethod
import re


class FrameFiller(ABC):
    def __init__(self, instruction):
        self.instruction = instruction

    @abstractmethod
    def fill(self):
        pass


class NaiveFrameFiller(FrameFiller):
    def fill(self):
        text = self.instruction.lower().strip(" .?!")

        match = re.search(r"bring me (?:the |a |an )?(.+?)(?: from|on (?:the )?(.+))?$", text)
        if not match:
            return None

        theme, source = match.groups()

        return {
            "Agent": "robot",
            "Theme": theme,
            "Source": source,
            "Destination": "user",
        }