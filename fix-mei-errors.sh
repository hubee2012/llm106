#!/bin/bash
# Fix frequent Intel MEI kernel errors:
#   mei_me: wait hw ready failed
#   mei_me: hw_start failed ret = -62 fw status = ...
#
# (Log lines like "mei melo / hu_start / wait hu ready" are the same mei_me timeout.)
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "请用 sudo 运行: sudo bash $0"
  exit 1
fi

CONF=/etc/modprobe.d/blacklist-mei.conf
cat > "$CONF" << 'EOF'
# Suppress Intel MEI timeout spam (ret=-62):
#   mei_me: wait hw ready failed
#   mei_me: hw_start failed ret = -62 fw status = ...
# Does not affect NVIDIA GPU. May disable HDCP on Intel iGPU HDMI.
blacklist mei_hdcp
blacklist mei_pxp
blacklist mei_me
blacklist mei
EOF

echo "已写入 $CONF"
cat "$CONF"
echo

modprobe -r mei_hdcp 2>/dev/null || true
modprobe -r mei_pxp 2>/dev/null || true
modprobe -r mei_me 2>/dev/null || true
modprobe -r mei 2>/dev/null || true

echo "当前 mei 模块:"
lsmod | grep mei || echo "(已全部卸载)"
echo

update-initramfs -u
echo
echo "完成。请执行: sudo reboot"
echo "重启后确认: journalctl -k -b | grep -i 'hw_start failed' || echo OK"
