import pytest

from cast_service.app import create_app
from cast_service.cache import TTLCache
from cast_service.providers import (MOVIE, SERIES, TMDB, TVmaze, ProviderError,
                                    normalise_title, pick_best)
from cast_service.service import CastService


class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


class FakeSession:
    """Routes by URL suffix; records calls."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return FakeResp(404, {})


TMDB_MOVIE_SEARCH = FakeResp(200, {"results": [
    {"id": 1, "title": "The Matrix Resurrections", "release_date": "2021-12-22"},
    {"id": 603, "title": "The Matrix", "release_date": "1999-03-31"},
]})
TMDB_MOVIE = FakeResp(200, {"title": "The Matrix", "release_date": "1999-03-31",
    "credits": {"cast": [
        {"id": 2, "name": "Carrie-Anne Moss", "character": "Trinity", "order": 2, "profile_path": "/c.jpg"},
        {"id": 1, "name": "Keanu Reeves", "character": "Neo", "order": 0, "profile_path": "/k.jpg"},
        {"id": 3, "name": "Extra", "character": "", "order": 9, "profile_path": None},
    ]}})
TMDB_TV_SEARCH = FakeResp(200, {"results": [{"id": 43, "name": "Line of Duty", "first_air_date": "2012-06-26"}]})
TMDB_TV = FakeResp(200, {"name": "Line of Duty", "first_air_date": "2012-06-26",
    "aggregate_credits": {"cast": [{"id": 9, "name": "Adrian Dunbar", "order": 0,
        "profile_path": "/a.jpg", "total_episode_count": 36,
        "roles": [{"character": "Cameo", "episode_count": 1},
                  {"character": "Ted Hastings", "episode_count": 36}]}]}})
TVMAZE_SEARCH = FakeResp(200, [{"show": {"id": 353, "name": "Line of Duty", "premiered": "2012-06-26"}}])
TVMAZE_SHOW = FakeResp(200, {"id": 353, "name": "Line of Duty", "premiered": "2012-06-26",
    "_embedded": {"cast": [
        {"person": {"id": 5, "name": "Adrian Dunbar", "image": {"medium": "http://p/a.jpg"}},
         "character": {"name": "Ted Hastings", "image": None}},
        {"person": {"id": 5, "name": "Adrian Dunbar", "image": None},
         "character": {"name": "Ted (young)"}},
        {"person": {"id": 6, "name": "Vicky McClure", "image": None},
         "character": {"name": "Kate Fleming", "image": {"medium": "http://c/k.jpg"}}},
    ]}})


def tmdb(routes):
    return TMDB("abc123", session=FakeSession(routes))


def tvmaze(routes):
    return TVmaze(session=FakeSession(routes))


# --- matching ---------------------------------------------------------------

@pytest.mark.parametrize("a,b", [("The Office", "Office"), ("Amélie", "Amelie"),
                                 ("Law & Order", "Law and Order"), ("Spider-Man", "spiderman")])
def test_normalise_title(a, b):
    assert normalise_title(a) == normalise_title(b)


def test_pick_best_prefers_exact_title_and_year():
    cands = [("Dune", 1984, "old"), ("Dune", 2021, "new"), ("Dune: Part Two", 2024, "two")]
    assert pick_best(cands, "Dune", 2021) == ("new", "exact")
    assert pick_best(cands, "Dune", None) == ("old", "exact")  # relevance order wins
    assert pick_best(cands, "Dune", 2022) == ("new", "exact")  # +-1 year tolerance
    assert pick_best([("Other", 2000, "x")], "Dune", None) == ("x", "fuzzy")
    assert pick_best([], "Dune", None) == (None, None)


# --- providers --------------------------------------------------------------

def test_tmdb_movie_search_and_cast():
    p = tmdb({"/search/movie": TMDB_MOVIE_SEARCH, "/movie/603": TMDB_MOVIE})
    sid, match = p.search("The Matrix", MOVIE, 1999)
    assert (sid, match) == (603, "exact")
    r = p.cast(sid, MOVIE)
    assert [c["name"] for c in r["cast"]] == ["Keanu Reeves", "Carrie-Anne Moss", "Extra"]
    assert r["cast"][0]["image"].endswith("/w185/k.jpg")
    assert r["cast"][2]["character"] is None and r["cast"][2]["image"] is None
    assert r["source"] == "tmdb" and "TMDB" in r["attribution"]


def test_tmdb_tv_uses_main_role_and_episode_count():
    p = tmdb({"/search/tv": TMDB_TV_SEARCH, "/tv/43": TMDB_TV})
    r = p.cast(p.search("Line of Duty", SERIES)[0], SERIES)
    assert r["cast"][0]["character"] == "Ted Hastings"
    assert r["cast"][0]["episode_count"] == 36


def test_tmdb_auth_modes():
    s = FakeSession({"/movie/1": TMDB_MOVIE})
    TMDB("eyJhbGciOi.token", session=s).cast(1, MOVIE)
    assert s.calls[-1][2]["Authorization"].startswith("Bearer ")
    assert "api_key" not in s.calls[-1][1]
    TMDB("0123abcd", session=s).cast(1, MOVIE)
    assert s.calls[-1][1]["api_key"] == "0123abcd"


def test_tmdb_errors():
    assert tmdb({}).cast(999, MOVIE) is None  # 404 -> None
    with pytest.raises(ProviderError):
        tmdb({"/search/movie": FakeResp(401, {})}).search("x", MOVIE)


def test_tvmaze_dedupes_and_prefers_headshot():
    p = tvmaze({"/search/shows": TVMAZE_SEARCH, "/shows/353": TVMAZE_SHOW})
    sid, _ = p.search("Line of Duty", SERIES, 2012)
    r = p.cast(sid, SERIES)
    assert [c["name"] for c in r["cast"]] == ["Adrian Dunbar", "Vicky McClure"]
    assert r["cast"][0]["image"] == "http://p/a.jpg"
    assert r["cast"][1]["image"] == "http://c/k.jpg"  # falls back to character still
    assert p.search("The Matrix", MOVIE) == (None, None)


# --- service ----------------------------------------------------------------

def test_series_falls_back_to_tvmaze_when_tmdb_errors():
    import requests
    svc = CastService(
        tmdb=tmdb({"/search/tv": requests.ConnectionError("down")}),
        tvmaze=tvmaze({"/search/shows": TVMAZE_SEARCH, "/shows/353": TVMAZE_SHOW}))
    r = svc.by_title("Line of Duty", SERIES)
    assert r["source"] == "tvmaze" and r["match"] == "exact"


def test_series_works_without_tmdb_key():
    svc = CastService(tvmaze=tvmaze({"/search/shows": TVMAZE_SEARCH, "/shows/353": TVMAZE_SHOW}))
    assert svc.by_title("Line of Duty", SERIES)["source"] == "tvmaze"
    with pytest.raises(LookupError):
        svc.by_title("The Matrix", MOVIE)


def test_caching_hits_and_misses():
    t = tmdb({"/search/movie": TMDB_MOVIE_SEARCH, "/movie/603": TMDB_MOVIE})
    svc = CastService(tmdb=t)
    svc.by_title("The Matrix", MOVIE, 1999)
    n = len(t.http.session.calls)
    assert svc.by_title("the matrix ", MOVIE, 1999, limit=1)["cast"][0]["name"] == "Keanu Reeves"
    assert svc.by_id("tmdb", MOVIE, 603)["source_id"] == 603  # warmed by title lookup
    assert len(t.http.session.calls) == n

    empty = tmdb({"/search/movie": FakeResp(200, {"results": []})})
    svc = CastService(tmdb=empty)
    assert svc.by_title("Nope", MOVIE) is None
    assert svc.by_title("Nope", MOVIE) is None
    assert len(empty.http.session.calls) == 1  # negative-cached


def test_errors_are_not_negative_cached():
    t = tmdb({"/search/movie": FakeResp(500, {})})
    svc = CastService(tmdb=t)
    for _ in range(2):
        with pytest.raises(ProviderError):
            svc.by_title("X", MOVIE)
    assert len(t.http.session.calls) == 2


def test_ttl_cache_expiry_and_eviction():
    now = [0.0]
    c = TTLCache(max_items=2, clock=lambda: now[0])
    c.set("a", 1, 10); c.set("b", 2, 10); c.set("c", 3, 10)
    assert c.get("a") is None and c.get("c") == 3
    now[0] = 11
    assert c.get("c") is None


# --- HTTP -------------------------------------------------------------------

@pytest.fixture
def client():
    svc = CastService(
        tmdb=tmdb({"/search/movie": TMDB_MOVIE_SEARCH, "/movie/603": TMDB_MOVIE,
                   "/search/tv": FakeResp(200, {"results": []})}),
        tvmaze=tvmaze({"/search/shows": TVMAZE_SEARCH, "/shows/353": TVMAZE_SHOW}))
    return create_app(svc).test_client()


def test_http_by_title(client):
    r = client.get("/v1/cast?title=The%20Matrix&type=film&year=1999&limit=2")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["cast"]) == 2 and body["match"] == "exact"
    assert "max-age" in r.headers["Cache-Control"]
    assert client.get("/v1/cast?title=Line%20of%20Duty&type=series").get_json()["source"] == "tvmaze"


def test_http_by_id_and_errors(client):
    assert client.get("/v1/cast/tmdb/movie/603").get_json()["match"] == "id"
    assert client.get("/v1/cast/tmdb/movie/1").status_code == 404
    assert client.get("/v1/cast/imdb/movie/1").status_code == 503
    assert client.get("/v1/cast?title=x").status_code == 400
    assert client.get("/v1/cast?type=movie").status_code == 400
    assert client.get("/v1/cast?title=x&type=movie&year=99").status_code == 400
    assert client.get("/v1/cast?title=x&type=movie&limit=abc").status_code == 400
    assert client.get("/health").get_json()["providers"] == ["tmdb", "tvmaze"]


def test_http_provider_down_is_502():
    svc = CastService(tmdb=tmdb({"/search/movie": FakeResp(503, {})}))
    assert create_app(svc).test_client().get("/v1/cast?title=x&type=movie").status_code == 502
