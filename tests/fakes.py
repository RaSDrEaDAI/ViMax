"""Test doubles shared across the suite. No test may reach a real model."""

from typing import Any, List


class FakeChatModel:
    """Stands in for a LangChain chat model. Each ``ainvoke`` returns the next
    scripted response (repeating the last one), so a test can script "bad
    answer, then good answer" and assert how many times the agent asked.

    Supports both call shapes agents use: ``with_structured_output(Schema)``
    and ``chat_model | parser`` (the parser is bypassed; responses are already
    the parsed objects).
    """

    def __init__(self, responses: List[Any]):
        self.responses = list(responses)
        self.calls = 0
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    def __or__(self, parser):
        return self

    async def ainvoke(self, messages, *args, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, BaseException):
            raise response
        return response
