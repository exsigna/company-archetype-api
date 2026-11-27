"""Cast lookups from TMDB (films + series) and TVmaze (series fallback).

Everything is normalised to one shape so the Fire TV app never needs to know
which provider answered:

    {"title", "type", "year", "source", "source_id", "match",
     "cast": [{"name", "character", "image", "person_id", "episode_count"}],
     "attribution"}
"""
import re
import unicodedata

import requests

MOVIE, SERIES = "movie", "series"

TMDB_API = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w185"  # ~185px wide: plenty for a TV cast row
TVMAZE_API = "https://api.tvmaze.com"

TMDB_ATTRIBUTION = ("This product uses the TMDB API but is not endorsed or "
                    "certified by TMDB.")
TVMAZE_ATTRIBUTION = "Cast data from TVmaze (tvmaze.com), CC BY-SA 4.0."


class ProviderError(Exception):
    """Upstream failed (network, 5xx, auth). Not cached."""


def normalise_title(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and")
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _year(date_str):
    return int(date_str[:4]) if date_str and date_str[:4].isdigit() else None


def pick_best(candidates, title, year):
    """candidates: list of (title, year, payload) in provider relevance order.

    Preference: exact title + year > exact title > year within 1 > first result.
    Returns (payload, match) where match is "exact" or "fuzzy", or (None, None).
    """
    if not candidates:
        return None, None
    want = normalise_title(title)
    exact = [c for c in candidates if normalise_title(c[0]) == want]
    if year:
        for c in exact:
            if c[1] == year:
                return c[2], "exact"
        # Remakes/reboots share titles; only accept a year-mismatched exact
        # title if it's within a year (release date differences by region).
        for c in exact:
            if c[1] and abs(c[1] - year) <= 1:
                return c[2], "exact"
        for c in candidates:
            if c[1] and abs(c[1] - year) <= 1:
                return c[2], "fuzzy"
        return (exact[0][2], "fuzzy") if exact else (candidates[0][2], "fuzzy")
    if exact:
        return exact[0][2], "exact"
    return candidates[0][2], "fuzzy"


class _Http:
    def __init__(self, session=None, timeout=5):
        self.session = session or requests.Session()
        self.timeout = timeout

    def get_json(self, url, params=None, headers=None, allow_404=False):
        try:
            r = self.session.get(url, params=params, headers=headers,
                                 timeout=self.timeout)
        except requests.RequestException as e:
            raise ProviderError(f"{url}: {e}") from e
        if allow_404 and r.status_code == 404:
            return None
        if r.status_code != 200:
            raise ProviderError(f"{url}: HTTP {r.status_code}")
        return r.json()


class TMDB:
    name = "tmdb"
    attribution = TMDB_ATTRIBUTION

    def __init__(self, api_key, language="en-GB", region="GB", **http_kw):
        if not api_key:
            raise ValueError("TMDB api_key required")
        self.http = _Http(**http_kw)
        self.language, self.region = language, region
        # v4 read-access tokens are JWTs; v3 keys are 32 hex chars.
        if api_key.startswith("eyJ"):
            self._headers, self._auth = {"Authorization": f"Bearer {api_key}"}, {}
        else:
            self._headers, self._auth = {}, {"api_key": api_key}

    def _get(self, path, allow_404=False, **params):
        params = {**self._auth, "language": self.language, **params}
        return self.http.get_json(TMDB_API + path, params=params,
                                  headers=self._headers, allow_404=allow_404)

    def search(self, title, kind, year=None):
        if kind == MOVIE:
            params = {"query": title, "region": self.region}
            data = self._get("/search/movie", **params)
            cands = [(r.get("title"), _year(r.get("release_date")), r)
                     for r in data.get("results", [])]
        else:
            data = self._get("/search/tv", query=title)
            cands = [(r.get("name"), _year(r.get("first_air_date")), r)
                     for r in data.get("results", [])]
        hit, match = pick_best(cands, title, year)
        return (hit["id"], match) if hit else (None, None)

    def cast(self, source_id, kind, limit=20):
        if kind == MOVIE:
            info = self._get(f"/movie/{source_id}", allow_404=True,
                             append_to_response="credits")
            if info is None:
                return None
            title, year = info.get("title"), _year(info.get("release_date"))
            people = (info.get("credits") or {}).get("cast", [])
            cast = [{
                "name": p.get("name"),
                "character": p.get("character") or None,
                "image": TMDB_IMG + p["profile_path"] if p.get("profile_path") else None,
                "person_id": p.get("id"),
                "episode_count": None,
            } for p in sorted(people, key=lambda p: p.get("order", 999))]
        else:
            info = self._get(f"/tv/{source_id}", allow_404=True,
                             append_to_response="aggregate_credits")
            if info is None:
                return None
            title, year = info.get("name"), _year(info.get("first_air_date"))
            people = (info.get("aggregate_credits") or {}).get("cast", [])
            cast = []
            for p in sorted(people, key=lambda p: p.get("order", 999)):
                # Main role = the one they played in most episodes.
                roles = sorted(p.get("roles") or [],
                               key=lambda r: -(r.get("episode_count") or 0))
                cast.append({
                    "name": p.get("name"),
                    "character": (roles[0].get("character") or None) if roles else None,
                    "image": TMDB_IMG + p["profile_path"] if p.get("profile_path") else None,
                    "person_id": p.get("id"),
                    "episode_count": p.get("total_episode_count"),
                })
        return {"title": title, "type": kind, "year": year, "source": self.name,
                "source_id": int(source_id), "cast": cast[:limit],
                "attribution": self.attribution}


class TVmaze:
    """Series only. Free, no key; rate limit ~20 calls / 10s per IP."""
    name = "tvmaze"
    attribution = TVMAZE_ATTRIBUTION

    def __init__(self, **http_kw):
        self.http = _Http(**http_kw)

    def search(self, title, kind, year=None):
        if kind != SERIES:
            return None, None
        data = self.http.get_json(f"{TVMAZE_API}/search/shows", params={"q": title})
        cands = [(r["show"].get("name"), _year(r["show"].get("premiered")), r["show"])
                 for r in data or []]
        hit, match = pick_best(cands, title, year)
        return (hit["id"], match) if hit else (None, None)

    def cast(self, source_id, kind, limit=20):
        if kind != SERIES:
            return None
        show = self.http.get_json(f"{TVMAZE_API}/shows/{source_id}",
                                  params={"embed": "cast"}, allow_404=True)
        if show is None:
            return None
        cast, seen = [], set()
        for c in (show.get("_embedded") or {}).get("cast", []):
            person = c.get("person") or {}
            # TVmaze lists an actor once per character; keep the first.
            if person.get("id") in seen:
                continue
            seen.add(person.get("id"))
            # Prefer the actor's headshot; fall back to the character still.
            img = person.get("image") or (c.get("character") or {}).get("image")
            cast.append({
                "name": person.get("name"),
                "character": (c.get("character") or {}).get("name"),
                "image": (img or {}).get("medium"),
                "person_id": person.get("id"),
                "episode_count": None,
            })
        return {"title": show.get("name"), "type": SERIES,
                "year": _year(show.get("premiered")), "source": self.name,
                "source_id": int(source_id), "cast": cast[:limit],
                "attribution": self.attribution}
