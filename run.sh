#!/usr/bin/env bash
# ==============================================================================
# SPIDER (System for Portfolio, Index, and Data Economic Relationships)
# Launch Script
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "============================================================"
echo " 🕷️  Avvio SPIDER (Macro & Market Network Monitor)"
echo "============================================================"

# Check Python command
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Errore: Python non trovato nel sistema."
    exit 1
fi

echo "Verifica dipendenze richieste..."
$PY_CMD -c "import streamlit, pandas, numpy, scipy, plotly, requests, openpyxl, certifi" 2>/dev/null || {
    echo "Installazione dipendenze mancanti da requirements.txt..."
    $PY_CMD -m pip install -r requirements.txt
}

echo "Rilevamento indirizzo di rete locale..."
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo "============================================================"
echo " 💻 Accesso da questo Mac:      http://localhost:8502"
echo " 📱 Accesso da Mobile (Wi-Fi):   http://${LOCAL_IP}:8502"
echo "============================================================"
echo ""
echo "Avvio motore analitico e interfaccia grafica SPIDER su porta 8502..."
$PY_CMD -m streamlit run app.py --server.port 8502 --server.address 0.0.0.0 --server.headless false
