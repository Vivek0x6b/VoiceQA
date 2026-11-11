"""Transcript schema shared by every STT engine and written as transcript.json."""

from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = []
    speaker: str | None = None
    # Set in scenario runs: the turn id from events.json ("agent-1", "caller-2").
    turn: str | None = None


class Transcript(BaseModel):
    segments: list[Segment] = []
    engine: str
    model: str
    language: str
