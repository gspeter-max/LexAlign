#!/usr/bin/env bash
#
# start.sh — Launch LexAlign frontend + backend with one command
#
# Usage:
#   ./start.sh          # Start both servers
#   ./start.sh --api    # Start API only
#   ./start.sh --ui     # Start frontend only
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
VIOLET='\033[0;35m'
RESET='\033[0m'
BOLD='\033[1m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${VIOLET}${BOLD}  ╔═══════════════════════════════════════╗${RESET}"
echo -e "${VIOLET}${BOLD}  ║          LexAlign Pipeline            ║${RESET}"
echo -e "${VIOLET}${BOLD}  ║   Download → Fine-Tune → Align        ║${RESET}"
echo -e "${VIOLET}${BOLD}  ╚═══════════════════════════════════════╝${RESET}"
echo ""

# ─── Cleanup on exit ──────────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${CYAN}Shutting down...${RESET}"
    kill $API_PID $UI_PID 2>/dev/null
    wait $API_PID $UI_PID 2>/dev/null
    echo -e "${GREEN}Done.${RESET}"
}
trap cleanup EXIT INT TERM

API_PID=""
UI_PID=""

# ─── Start API ────────────────────────────────────────────────
start_api() {
    echo -e "${CYAN}[API]${RESET} Starting FastAPI backend on http://localhost:8000 ..."

    if [ ! -d "$PROJECT_DIR/.venv" ]; then
        echo -e "${RED}[API]${RESET} No .venv found. Run: python -m venv .venv && pip install -e '.[dev]'"
        exit 1
    fi

    "$PROJECT_DIR/.venv/bin/uvicorn" api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info &
    API_PID=$!

    echo -e "${GREEN}[API]${RESET} Backend started (PID: $API_PID)"
}

# ─── Start Frontend ──────────────────────────────────────────
start_ui() {
    echo -e "${CYAN}[UI]${RESET}  Starting Next.js frontend on http://localhost:3000 ..."

    if [ ! -d "$PROJECT_DIR/frontend/node_modules" ]; then
        echo -e "${CYAN}[UI]${RESET}  Installing Node dependencies..."
        (cd "$PROJECT_DIR/frontend" && npm install)
    fi

    (cd "$PROJECT_DIR/frontend" && npm run dev) &
    UI_PID=$!

    echo -e "${GREEN}[UI]${RESET}  Frontend started (PID: $UI_PID)"
}

# ─── Main ─────────────────────────────────────────────────────
case "${1:-}" in
    --api)
        start_api
        wait $API_PID
        ;;
    --ui)
        start_ui
        wait $UI_PID
        ;;
    *)
        start_api
        start_ui

        echo ""
        echo -e "${GREEN}${BOLD}  ✓ LexAlign is running!${RESET}"
        echo -e "    ${CYAN}Frontend:${RESET} http://localhost:3000"
        echo -e "    ${CYAN}API:${RESET}      http://localhost:8000"
        echo -e "    ${CYAN}API Docs:${RESET} http://localhost:8000/docs"
        echo ""
        echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop."
        echo ""

        wait
        ;;
esac
