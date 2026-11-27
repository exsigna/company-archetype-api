"""Small thread-safe in-memory TTL cache.

In-memory is enough for a single process. With several gunicorn workers each
keeps its own copy; switch to Redis if hit rate or provider rate limits matter.
"""
import threading
import time
from collections import OrderedDict

_MISSING = object()


class TTLCache:
    def __init__(self, max_items=5000, clock=time.monotonic):
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_items
        self._clock = clock

    def get(self, key, default=_MISSING):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None if default is _MISSING else default
            expires, value = item
            if expires < self._clock():
                del self._data[key]
                return None if default is _MISSING else default
            self._data.move_to_end(key)
            return value

    def set(self, key, value, ttl):
        with self._lock:
            self._data[key] = (self._clock() + ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __len__(self):
        return len(self._data)
