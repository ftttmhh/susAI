@echo off
REM Launch backend (FastAPI) and frontend (React) together

REM Start backend
start cmd /k "cd backend && .venv\Scripts\activate && uvicorn main:app --reload --port 8000"

REM Start frontend
start cmd /k "cd frontend && npm start"

REM This will open two new terminal windows: one for backend, one for frontend.
