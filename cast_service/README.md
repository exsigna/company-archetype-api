# TV+ cast service

Backend for the Fire TV app's cast panel. It looks up cast for **films and series**, gives every answer the same shape, and caches the results. The sticks call this service, never the providers directly, so API keys stay off the devices.

| Content | Primary | Fallback |
|---|---|---|
| Films | TMDB (needs `TMDB_API_KEY`) | none |
| Series | TMDB | TVmaze (free, no key) |

Without a TMDB key, series still work through TVmaze and films return `503`.

## Run

```bash
pip install -r cast_service/requirements.txt
export TMDB_API_KEY=...        # a v3 key or a v4 read-access token both work
gunicorn 'cast_service.app:create_app()'
```

## API

```
GET /v1/cast?title=Line%20of%20Duty&type=series&year=2012&limit=20
GET /v1/cast/tmdb/movie/603          # by provider ID: exact, so prefer it
GET /v1/cast/tvmaze/series/353
GET /health
```

`type` is `movie` or `series`. `year` is optional but tells remakes apart (for example *The Office* UK 2001 and US 2005). `limit` runs from 1 to 50 and defaults to 20.

```json
{
  "title": "Line of Duty", "type": "series", "year": 2012,
  "source": "tvmaze", "source_id": 353, "match": "exact",
  "cast": [{"name": "Adrian Dunbar", "character": "Superintendent Edward \"Ted\" Hastings",
            "image": "https://...jpg", "person_id": 41585, "episode_count": null}],
  "attribution": "Cast data from TVmaze (tvmaze.com), CC BY-SA 4.0."
}
```

- `match` is `exact` (the title matched, and the year too if one was given), `fuzzy` (a best guess, so consider hiding the cast panel) or `id` (looked up by ID).
- `image` and `character` can be `null`, so show a placeholder when they are.
- `episode_count` is filled for TMDB series only.
- Errors: `400` for bad parameters, `404` when no cast was found, `502` when the provider is down, and `503` when no provider is set up for that type.

## Must-dos before release

- **TMDB licence:** a paid or ad-funded app needs a commercial agreement with TMDB. You also have to show the `attribution` text and the TMDB logo somewhere in the app, such as an About screen.
- **TVmaze:** credit it under CC BY-SA by showing the `attribution` string.
- **Caching:** results are held in memory for 7 days, and "not found" for 1 day. Each gunicorn worker keeps its own cache, so move to Redis if you run several workers or instances.

## Tests

```bash
pytest cast_service
```
