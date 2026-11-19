#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! python3 -c "import numpy, pandas, matplotlib, docx, streamlit" >/dev/null 2>&1; then
    echo "Установка необходимых библиотек..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Запуск веб-интерфейса Spinel Studio..."
python3 -m streamlit run app.py
