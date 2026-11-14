#!/usr/bin/env bash
# Run app_web.py on a Colab T4 and print the public gradio.live link.
# Stop with: colab stop -s darkir
set -euo pipefail
cd "$(dirname "$0")"
S=darkir
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

colab new -s $S --gpu T4

# Colab upload fails on big files, so send 4MB chunks.
tar czf "$T/d.tgz" --exclude=./venv_DarkIR --exclude=./.git --exclude=__pycache__ \
  --exclude=./videos --exclude=./images --exclude=./assets --exclude=./app.py \
  --exclude=./models/EnlightenGAN_200_net_G_A.pth \
  . assets/qualis/inputs/0010.png assets/qualis/inputs/0075.png
(cd "$T" && split -b 4M d.tgz part_ && for p in part_*; do colab upload -s $S "$p" "/content/$p"; done)

echo 'import subprocess;subprocess.run("cd /content && cat part_* > d.tgz && rm part_* && mkdir -p DarkIR && tar xzf d.tgz -C DarkIR",shell=True,check=True)' \
  | colab exec -s $S --timeout 300
colab install -s $S gradio==5.12.0 gradio_client==1.5.4 huggingface_hub==0.36.2 ptflops==0.7.3

colab exec -s $S --timeout 240 <<'EOF'
import subprocess, time
subprocess.Popen("cd /content/DarkIR && PYTHONUNBUFFERED=1 nohup python app_web.py --share > /content/app.log 2>&1 &", shell=True)
for _ in range(90):
    time.sleep(2)
    log = open("/content/app.log").read()
    if "gradio.live" in log or "Traceback" in log:
        break
print(log[-2000:])
EOF
