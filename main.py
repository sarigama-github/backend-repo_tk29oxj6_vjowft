import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        # Try to import database module
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # noqa: BLE001
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:  # noqa: BLE001
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ---------------------------
# Weights & Biases Proxy API
# ---------------------------
WANDB_GQL_URL = "https://api.wandb.ai/graphql"


def _wandb_headers() -> Dict[str, str]:
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Server is missing WANDB_API_KEY env var")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


@app.get("/api/wandb/runs")
def list_wandb_runs(
    entity: str = Query(..., description="W&B entity/org/user"),
    project: str = Query(..., description="W&B project name"),
    limit: int = Query(25, ge=1, le=200, description="Max runs to return"),
):
    """Return recent runs for a given entity/project with key summary metrics.

    This proxies W&B GraphQL to keep the API key on the server.
    """
    headers = _wandb_headers()

    # GraphQL query to fetch project runs and summary metrics
    query = """
    query ProjectRuns($entity: String!, $project: String!, $first: Int!) {
      project(name: $project, entityName: $entity) {
        id
        name
        entity { name }
        runs(first: $first, order: {by: UPDATED_AT, direction: DESC}) {
          edges {
            node {
              id
              name
              displayName
              state
              createdAt
              updatedAt
              user { name }
              notes
              tags
              summaryMetrics
            }
          }
        }
      }
    }
    """

    payload = {"query": query, "variables": {"entity": entity, "project": project, "first": limit}}

    try:
        resp = requests.post(WANDB_GQL_URL, json=payload, headers=headers, timeout=20)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"W&B API error: {resp.text[:200]}")
        data = resp.json()
        if "errors" in data:
            # Flatten common error message
            message = "; ".join([e.get("message", "Unknown error") for e in data.get("errors", [])])
            raise HTTPException(status_code=400, detail=f"W&B GraphQL error: {message}")

        project_data = (data.get("data", {}) or {}).get("project")
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found or access denied")

        edges: List[Dict[str, Any]] = project_data.get("runs", {}).get("edges", [])
        runs: List[Dict[str, Any]] = []
        for e in edges:
            n = e.get("node", {})
            # summaryMetrics can be a JSON string or dict depending on API version
            summary = n.get("summaryMetrics")
            if isinstance(summary, str):
                try:
                    import json as _json

                    summary = _json.loads(summary)
                except Exception:
                    summary = None
            # Keep only a few numeric metrics to reduce payload
            slim_metrics: Dict[str, Any] = {}
            if isinstance(summary, dict):
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        slim_metrics[k] = v
                # trim to first 12 metrics
                slim_metrics = dict(list(slim_metrics.items())[:12])

            runs.append(
                {
                    "id": n.get("id"),
                    "name": n.get("name"),
                    "displayName": n.get("displayName") or n.get("name"),
                    "state": n.get("state"),
                    "createdAt": n.get("createdAt"),
                    "updatedAt": n.get("updatedAt"),
                    "user": (n.get("user") or {}).get("name"),
                    "notes": n.get("notes"),
                    "tags": n.get("tags") or [],
                    "metrics": slim_metrics,
                }
            )

        return {"project": project_data.get("name"), "entity": project_data.get("entity", {}).get("name"), "count": len(runs), "runs": runs}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Server error contacting W&B: {str(e)[:200]}") from e


@app.get("/api/wandb/run")
def get_wandb_run(
    entity: str = Query(...), project: str = Query(...), run: str = Query(..., description="Run name/ID")
):
    """Return details for a single run, including summary metrics.

    Note: For full history series, prefer W&B direct links. This keeps payloads light for mobile.
    """
    headers = _wandb_headers()

    query = """
    query Run($entity: String!, $project: String!, $run: String!) {
      project(name: $project, entityName: $entity) {
        run(name: $run) {
          id
          name
          displayName
          state
          createdAt
          updatedAt
          notes
          tags
          summaryMetrics
          historyKeys
        }
      }
    }
    """
    payload = {"query": query, "variables": {"entity": entity, "project": project, "run": run}}
    try:
        resp = requests.post(WANDB_GQL_URL, json=payload, headers=headers, timeout=20)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"W&B API error: {resp.text[:200]}")
        data = resp.json()
        if "errors" in data:
            message = "; ".join([e.get("message", "Unknown error") for e in data.get("errors", [])])
            raise HTTPException(status_code=400, detail=f"W&B GraphQL error: {message}")
        proj = (data.get("data", {}) or {}).get("project")
        if not proj or not proj.get("run"):
            raise HTTPException(status_code=404, detail="Run not found")
        run_node = proj["run"]
        summary = run_node.get("summaryMetrics")
        if isinstance(summary, str):
            try:
                import json as _json

                summary = _json.loads(summary)
            except Exception:
                summary = None
        return {"run": run_node, "summary": summary}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Server error contacting W&B: {str(e)[:200]}") from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
