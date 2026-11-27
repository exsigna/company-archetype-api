"""Lookup orchestration: provider order, fallback and caching."""
from .cache import TTLCache
from .providers import MOVIE, SERIES, ProviderError

HIT_TTL = 7 * 24 * 3600   # cast lists rarely change
MISS_TTL = 24 * 3600      # retry unknown titles daily (new releases appear)
_NOT_FOUND = {"_not_found": True}


class CastService:
    def __init__(self, tmdb=None, tvmaze=None, cache=None):
        self.providers = {p.name: p for p in (tmdb, tvmaze) if p}
        self.tmdb, self.tvmaze = tmdb, tvmaze
        self.cache = cache or TTLCache()

    def _chain(self, kind):
        if kind == MOVIE:
            return [p for p in (self.tmdb,) if p]
        return [p for p in (self.tmdb, self.tvmaze) if p]

    def by_title(self, title, kind, year=None, limit=20):
        """Returns result dict, or None if no provider knows the title.

        Raises ProviderError only if every provider in the chain errored.
        """
        chain = self._chain(kind)
        if not chain:
            raise LookupError(f"no provider configured for {kind}")
        key = ("title", kind, title.strip().lower(), year)
        cached = self.cache.get(key)
        if cached is not None:
            return None if cached is _NOT_FOUND else _trim(cached, limit)

        errors = []
        for p in chain:
            try:
                source_id, match = p.search(title, kind, year)
                if source_id is None:
                    continue
                result = p.cast(source_id, kind, limit=100)
                if result and result["cast"]:
                    result["match"] = match
                    self.cache.set(key, result, HIT_TTL)
                    self.cache.set(("id", p.name, kind, str(source_id)), result, HIT_TTL)
                    return _trim(result, limit)
            except ProviderError as e:
                errors.append(e)
        if errors and len(errors) == len(chain):
            raise ProviderError("; ".join(map(str, errors)))
        # Only cache "not found" if no provider errored; otherwise one flaky
        # provider would hide a title for a day.
        if not errors:
            self.cache.set(key, _NOT_FOUND, MISS_TTL)
        return None

    def by_id(self, source, kind, source_id, limit=20):
        p = self.providers.get(source)
        if p is None:
            raise LookupError(f"provider {source!r} not configured")
        key = ("id", source, kind, str(source_id))
        cached = self.cache.get(key)
        if cached is not None:
            return None if cached is _NOT_FOUND else _trim(cached, limit)
        result = p.cast(source_id, kind, limit=100)
        if result:
            result["match"] = "id"
        self.cache.set(key, result or _NOT_FOUND, HIT_TTL if result else MISS_TTL)
        return _trim(result, limit) if result else None


def _trim(result, limit):
    return {**result, "cast": result["cast"][:limit]}
