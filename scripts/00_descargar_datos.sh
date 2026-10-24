#!/usr/bin/env bash
# Descarga los datos de la línea 1A con los scripts del repositorio de Stefan Jansen
# (Machine Learning for Trading, 3.ª ed.). Uso: bash scripts/00_descargar_datos.sh
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
ML4T_REPO="${ML4T_REPO:-$RAIZ/../ml4t/machine-learning-for-trading}"
if [ ! -d "$ML4T_REPO" ]; then
  git clone --depth 1 https://github.com/stefan-jansen/machine-learning-for-trading.git "$ML4T_REPO"
fi
cd "$ML4T_REPO"
export PYTHONPATH="$ML4T_REPO" ML4T_DATA_PATH="$ML4T_REPO/data"
python data/etfs/market/download.py                                          # 100 ETFs Yahoo 2006-2025
python data/futures/positioning/cot_download.py --products ES --start-year 2017 --end-year 2025  # CFTC TFF
# El índice S&P 500 (sp500.csv) y las barras diarias AlgoSeek de componentes vienen con el repositorio.
# FRED (DFF, T10Y2Y, VIXCLS), COT 2006-2016 y el calendario FOMC los descarga/lee linea1a.datos.
# Con FRED_API_KEY, `python data/macro/download_alfred.py` deja VIXCLS/DGS de primera publicación y se usan.
