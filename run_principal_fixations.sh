#!/usr/bin/env bash
# Corrida principal pre-registrada (results/v2/fixations/PREREGISTRO.md), reanudable.
# La lanza el servicio de usuario memoria-fixations.service: si se apaga el equipo, retoma al
# iniciar sesión (se pierde a lo más la imagen en curso). Al terminar, el servicio se deshabilita.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
{
  echo "== inicio $(date '+%F %T')"
  ./venv/bin/python -u experiment_fixations.py run --split test \
      --images-file results/v2/fixations/test_images_300.txt --etas 2
  echo "== análisis $(date '+%F %T')"
  ./venv/bin/python -u experiment_fixations.py analyze --split test
  echo "== fin $(date '+%F %T')"
} >> logs/principal_fixations.log 2>&1
systemctl --user disable memoria-fixations.service || true
