#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data/mnist"
TMP_DIR="$(mktemp -d)"
ZIP_FILE="$TMP_DIR/mnist-dataset.zip"
EXTRACT_DIR="$TMP_DIR/extracted"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

REQUIRED_FILES=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

MIN_SIZES=(
    47040016
    60008
    7840016
    10008
)

mkdir -p "$DATA_DIR"

all_files_valid() {
    local i file size min_size
    for i in "${!REQUIRED_FILES[@]}"; do
        file="$DATA_DIR/${REQUIRED_FILES[$i]}"
        min_size="${MIN_SIZES[$i]}"
        if [[ ! -f "$file" ]] || [[ $(stat -c%s "$file") -lt "$min_size" ]]; then
            return 1
        fi
    done
    return 0
}

if all_files_valid; then
    echo "MNIST ya está disponible en $DATA_DIR"
    exit 0
fi

echo "Descargando MNIST desde Kaggle (hojjatk/mnist-dataset)..."

if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    curl -fL -H "Authorization: Bearer ${KAGGLE_API_TOKEN}" \
        -o "$ZIP_FILE" "$KAGGLE_URL"
elif [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  if command -v jq >/dev/null 2>&1; then
    KAGGLE_USER="$(jq -r .username "${HOME}/.kaggle/kaggle.json")"
    KAGGLE_KEY="$(jq -r .key "${HOME}/.kaggle/kaggle.json")"
  else
    KAGGLE_USER="$(python3 -c "import json; print(json.load(open('${HOME}/.kaggle/kaggle.json'))['username'])")"
    KAGGLE_KEY="$(python3 -c "import json; print(json.load(open('${HOME}/.kaggle/kaggle.json'))['key'])")"
  fi
  curl -fL -u "${KAGGLE_USER}:${KAGGLE_KEY}" \
      -o "$ZIP_FILE" "$KAGGLE_URL"
else
    echo "Error: se requiere autenticación de Kaggle." >&2
    echo "" >&2
    echo "Opciones:" >&2
    echo "  1. Crear un token en https://www.kaggle.com/settings y exportar:" >&2
    echo "     export KAGGLE_API_TOKEN=<tu_token>" >&2
    echo "  2. Colocar credenciales en ~/.kaggle/kaggle.json:" >&2
    echo '     {"username":"<usuario>","key":"<api_key>"}' >&2
    echo "     chmod 600 ~/.kaggle/kaggle.json" >&2
    exit 1
fi

if [[ ! -s "$ZIP_FILE" ]]; then
    echo "Error: la descarga está vacía. Revise las credenciales de Kaggle." >&2
    exit 1
fi

if ! unzip -tq "$ZIP_FILE" >/dev/null 2>&1; then
    echo "Error: el archivo descargado no es un ZIP válido." >&2
    echo "       Suele indicar credenciales incorrectas o falta de aceptar las reglas del dataset en Kaggle." >&2
    exit 1
fi

mkdir -p "$EXTRACT_DIR"
unzip -qo "$ZIP_FILE" -d "$EXTRACT_DIR"

for i in "${!REQUIRED_FILES[@]}"; do
    file="${REQUIRED_FILES[$i]}"
    min_size="${MIN_SIZES[$i]}"
    found="$(find "$EXTRACT_DIR" -name "$file" -type f | head -n 1 || true)"

    if [[ -z "$found" ]]; then
        echo "Error: no se encontró $file dentro del ZIP." >&2
        exit 1
    fi

    cp "$found" "$DATA_DIR/$file"
    size="$(stat -c%s "$DATA_DIR/$file")"

    if [[ "$size" -lt "$min_size" ]]; then
        echo "Error: $file parece corrupto ($size bytes, mínimo esperado: $min_size)." >&2
        exit 1
    fi

    echo "OK: $file ($size bytes)"
done

echo "MNIST listo en $DATA_DIR"
