import asyncio
import random
from string import ascii_letters
from typing import Optional


async def acompletion(messages: list[str]) -> str:
    """Simulate an asynchronous completion by generating a new message based on the input messages."""
    await asyncio.sleep(0.2)
    data = [msg + random.choice(ascii_letters) + str(random.randint(0, 5)) for msg in messages]
    return data[-1]


async def tool_call(message: str, **kwargs):
    """Simulate an asynchronous tool call that takes a message and some keyword arguments, and returns a response after a delay."""
    await asyncio.sleep(0.2)
    print(f"Tool called with message: {message}, kwargs: {kwargs}")
    args_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return f"Tool called with args: {args_repr}"


class CallBack:
    def __init__(self, message: str):
        self._message = message
        self._response: Optional[str] = None

    @property
    def message(self):
        return self._message

    @property
    def ready(self):
        return self._response is not None

    def set_response(self, response: str):
        self._response = response

    def __call__(self, *args, **kwargs) -> Optional[str]:
        # print(f"Callback called with message: {self.message}, args: {args}, kwargs: {kwargs}")
        if self._response is not None:
            print(f"Returning response: {self._response}")
            return self._response
        return None


class MessageCallback(CallBack):
    pass


class ToolCallback(CallBack):
    pass


class AgenticLoop:
    def __init__(self, message: str, total: int = 8):
        self.messages: list[str] = []
        self.messages.append(message)
        self.count = 0
        self.total = total
        self.callbacks = []

    def __aiter__(self):
        return self

    async def __anext__(self):

        if self.count >= self.total:
            raise StopAsyncIteration

        for cb in self.callbacks:
            if cb.ready:
                response = cb()
                print(f"Received callback response: {response}")
                self.messages.append(response)
            else:
                print("Callback not ready yet.")
                raise StopAsyncIteration

        print(f"Current count: {self.count}, total: {self.total}")
        print(f"Current messages: {self.messages}")
        self.count += 1
        response = await acompletion(self.messages)
        print(f"Received response: {response}")
        self.messages.append(response)

        if self.count % 2 == 0:
            cb = ToolCallback(response)
            self.callbacks.append(cb)
            return cb

        cb = MessageCallback(response)
        self.callbacks.append(cb)
        return cb


async def main():
    print("Starting agent stream...")
    count = 0

    async for msg in AgenticLoop("Initial reading", total=4):
        count += 1
        if isinstance(msg, MessageCallback):
            print(f"Processing message callback: {msg.message}")
            # Simulate setting a response after some processing
            msg.set_response(f"Processed: {msg.message}")
        elif isinstance(msg, ToolCallback):
            print(f"Processing tool callback: {msg.message}")
            res = await tool_call(msg.message, tool_arg="example", count=count)
            print(f"Tool call result: {res}")
            msg.set_response(res)

    print("Stream finished.")


asyncio.run(main())
