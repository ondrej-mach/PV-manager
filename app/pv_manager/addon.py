from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .frontend.templates import render_index
from .state import AppContext


def _mount_static(app: FastAPI) -> None:
    static_root = Path(__file__).resolve().parent / "frontend" / "static"
    app.mount("/static", StaticFiles(directory=static_root), name="static")


def _build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(render_index())

    @router.get("/api/status")
    async def get_status() -> JSONResponse:
        ha_error = ctx.get_ha_error()
        payload: dict[str, Any] = {
            "ok": ha_error is None,
            "snapshot_available": False,
            "training": ctx.get_training_status().as_dict(),
            "home_assistant_error": ha_error,
            "cycle_running": ctx.is_cycle_running(),
        }

        snapshot = await ctx.get_snapshot()
        if snapshot:
            payload.update(
                {
                    "snapshot_available": True,
                    "last_updated": snapshot.timestamp.isoformat(),
                    "interval_minutes": snapshot.interval_minutes,
                    "horizon_hours": snapshot.horizon_hours,
                    "summary": snapshot.summary,
                }
            )
        else:
            payload.update({"last_updated": None, "summary": None})

        return JSONResponse(payload)

    @router.get("/api/forecast")
    async def forecast() -> JSONResponse:
        snapshot = await ctx.get_snapshot()
        if snapshot is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Forecast not ready")
        return JSONResponse(snapshot.as_payload())

    @router.post("/api/training")
    async def trigger_training() -> JSONResponse:
        try:
            await ctx.trigger_training()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return JSONResponse({"started": True})

    @router.post("/api/cycle")
    async def trigger_cycle() -> JSONResponse:
        try:
            await ctx.trigger_cycle()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return JSONResponse({"started": True})

    @router.get("/api/settings")
    async def get_settings() -> JSONResponse:
        payload = await ctx.get_settings_payload()
        return JSONResponse(payload)

    @router.post("/api/settings/inverter")
    async def update_inverter(payload: dict[str, Any]) -> JSONResponse:
        try:
            updated = await ctx.update_inverter_settings(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(updated)

    @router.post("/api/settings/battery")
    async def update_battery(payload: dict[str, Any]) -> JSONResponse:
        try:
            updated = await ctx.update_battery_settings(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(updated)

    @router.post("/api/settings/pricing")
    async def update_pricing(payload: dict[str, Any]) -> JSONResponse:
        try:
            updated = await ctx.update_pricing_settings(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(updated)

    return router


def create_application() -> FastAPI:
    app = FastAPI(title="PV Manager", docs_url=None, redoc_url=None)
    ctx = AppContext()
    app.state.ctx = ctx
    _mount_static(app)
    app.include_router(_build_router(ctx))

    @app.on_event("startup")
    async def _startup() -> None:
        await ctx.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await ctx.stop()

    return app
