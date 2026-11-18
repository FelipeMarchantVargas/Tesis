#!/usr/bin/env bash
# Descarga el código y los pesos pre-entrenados de los modelos de saliencia.
# Requiere: git y gdown (pip install gdown). Ejecutar desde la raíz del repo.
set -euo pipefail
mkdir -p models third_party
gdown -q 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O models/u2net.pth            # U2-Net (xuebinqin/U-2-Net)

cd third_party
[ -d TranSalNet ] || git clone -q --depth 1 https://github.com/LJOVO/TranSalNet
[ -d MobileSal ]  || git clone -q --depth 1 https://github.com/yuhuan-wu/MobileSal
mkdir -p TranSalNet/pretrained_models MobileSal/pretrained
gdown -q 14czAAQQcRLGeiddPOM6AaTJTieu6QiHy -O TranSalNet/pretrained_models/TranSalNet_Res.pth
gdown -q 1JVTYq5UE6Q0OHoOVoXWF5WW5w42jlM1T -O TranSalNet/pretrained_models/TranSalNet_Dense.pth
gdown -q 1AdTljzE3tvTIgTxWCEdf0g9ZWt68RCVD -O TranSalNet/pretrained_models/resnet50-0676ba61.pth
gdown -q 1IZ8EtoM7Ui8QA_MlX7lqcIhusLa3ddl6 -O TranSalNet/pretrained_models/densenet161-8d451a50.pth
gdown -q 1dfyFkdsI1rOfmhmgG-o45ggnOj5Wpr1d -O MobileSal/pretrained/mobilesal_ss.pth   # single-scale, NJU2K+NLPR
echo "Modelos listos."
