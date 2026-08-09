.PHONY: help prefetch up down logs ps migrate revision \
        dev-api dev-worker dev-ui test test-integration typecheck qdrant

help:
	@echo "Docker (recommended):"
	@echo "  make prefetch     one-time, ONLINE: download models into model-cache volume"
	@echo "  make up           build + start qdrant/redis/api/worker/ui (offline)"
	@echo "  make down         stop and remove containers"
	@echo "  make logs         tail all logs"
	@echo ""
	@echo "Local dev (bare processes in the uv venv; needs local redis + qdrant):"
	@echo "  make qdrant       run just Qdrant in Docker on :6333"
	@echo "  make dev-api      uvicorn service_embed.api:app --reload --port 8001"
	@echo "  make dev-worker   celery -A service_embed.celery_app worker"
	@echo "  make dev-ui       streamlit run ui/app.py --server.port 8502"
	@echo ""
	@echo "DB migrations (SQLAlchemy + Alembic):"
	@echo "  make migrate                 alembic upgrade head"
	@echo "  make revision m='message'    autogenerate a new migration"
	@echo ""
	@echo "  make test / make test-integration / make typecheck"

# --- Docker ---------------------------------------------------------------
prefetch:
	# --build is required: the prefetch service sits behind a compose profile, so
	# a plain `docker compose build` skips it and would run a stale image.
	docker compose --profile prefetch run --rm --build prefetch

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

qdrant:
	docker compose up -d qdrant

# --- Local dev (uv venv) --------------------------------------------------
dev-api:
	uv run uvicorn service_embed.api:app --reload --port 8001

dev-worker:
	uv run celery -A service_embed.celery_app worker --loglevel=info --concurrency=2

dev-ui:
	uv run streamlit run ui/app.py --server.port 8502

# --- Migrations -----------------------------------------------------------
migrate:
	uv run alembic upgrade head

revision:
	uv run alembic revision --autogenerate -m "$(m)"

# --- Quality --------------------------------------------------------------
test:
	uv run pytest

# Opt-in: needs a real Qdrant. `make qdrant` first, or point the var elsewhere.
test-integration:
	RAG_EMBED_TEST_QDRANT_URL=$${RAG_EMBED_TEST_QDRANT_URL:-http://localhost:6333} \
		uv run pytest -m integration -v

typecheck:
	uv run mypy --strict rag_embedding service_embed ui/app.py tests
