"""HTTP API for the TV+ Fire TV app's cast panel.

GET /v1/cast?title=Line%20of%20Duty&type=series[&year=2012][&limit=20]
GET /v1/cast/tmdb/movie/603          (preferred once you store provider IDs)
GET /health

Run:  TMDB_API_KEY=... gunicorn 'cast_service.app:create_app()'
"""
import os

from flask import Flask, jsonify, request

from .providers import MOVIE, SERIES, TMDB, TVmaze, ProviderError
from .service import CastService

KINDS = {"movie": MOVIE, "film": MOVIE, "series": SERIES, "tv": SERIES, "show": SERIES}
MAX_LIMIT = 50


def create_app(service=None):
    app = Flask(__name__)
    if service is None:
        key = os.environ.get("TMDB_API_KEY")
        service = CastService(tmdb=TMDB(key) if key else None, tvmaze=TVmaze())
    app.config["CAST_SERVICE"] = service

    def err(status, msg):
        return jsonify({"error": msg}), status

    def parse_limit():
        try:
            return min(max(int(request.args.get("limit", 20)), 1), MAX_LIMIT)
        except ValueError:
            return None

    def respond(fn):
        try:
            result = fn()
        except LookupError as e:
            return err(503, str(e))
        except ProviderError:
            app.logger.exception("cast provider failure")
            return err(502, "cast provider unavailable, try again later")
        if result is None:
            return err(404, "no cast found")
        resp = jsonify(result)
        # Let the stick / any CDN cache too; cast lists are near-static.
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    @app.get("/v1/cast")
    def cast_by_title():
        kind = KINDS.get((request.args.get("type") or "").lower())
        if not kind:
            return err(400, "type must be 'movie' or 'series'")
        limit = parse_limit()
        if limit is None:
            return err(400, "limit must be an integer")
        title = (request.args.get("title") or "").strip()
        if not title:
            return err(400, "title is required")
        year = request.args.get("year")
        if year and not (year.isdigit() and len(year) == 4):
            return err(400, "year must be YYYY")
        return respond(lambda: service.by_title(title, kind, int(year) if year else None, limit))

    @app.get("/v1/cast/<source>/<kind_s>/<int:source_id>")
    def cast_by_id(source, kind_s, source_id):
        kind = KINDS.get(kind_s.lower())
        if not kind:
            return err(400, "type must be 'movie' or 'series'")
        limit = parse_limit()
        if limit is None:
            return err(400, "limit must be an integer")
        return respond(lambda: service.by_id(source, kind, source_id, limit))

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "providers": sorted(service.providers)})

    return app
