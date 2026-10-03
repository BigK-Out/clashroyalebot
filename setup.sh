#!/usr/bin/env bash
# One-shot dev environment setup: Android SDK + emulator (AVD "cr"), scrcpy, v4l2loopback.
# Safe to re-run.
set -euo pipefail

SDK="$HOME/Android/Sdk"
CMDLINE_ZIP="commandlinetools-linux-16111833_latest.zip"
SYSIMG="system-images;android-30;google_apis;x86_64"  # API 30: last Google image with ARM translation (libndk_translation)
AVD_NAME="cr"
SCRCPY_DIR="$HOME/.local/opt/scrcpy"

echo "==> apt packages"
sudo apt update
sudo apt install -y v4l-utils ffmpeg openjdk-17-jre-headless unzip curl libclang-dev  # libclang: v4l crate bindgen

echo "==> Android cmdline-tools"
if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
  mkdir -p "$SDK/cmdline-tools"
  tmp=$(mktemp -d)
  curl -fL "https://dl.google.com/android/repository/$CMDLINE_ZIP" -o "$tmp/clt.zip"
  unzip -q "$tmp/clt.zip" -d "$tmp"
  rm -rf "$SDK/cmdline-tools/latest"
  mv "$tmp/cmdline-tools" "$SDK/cmdline-tools/latest"
  rm -rf "$tmp"
fi

export ANDROID_HOME="$SDK"
export PATH="$SDK/cmdline-tools/latest/bin:$SDK/platform-tools:$SDK/emulator:$PATH"

if ! grep -q "ANDROID_HOME" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# Android SDK
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$HOME/.local/bin:$PATH
EOF
fi

echo "==> SDK packages (platform-tools, emulator, system image)"
# Parallel range downloads: sdkmanager's single connection to dl.google.com is throttled.
python3 "$(dirname "$0")/scripts/fetch_sdk.py"

echo "==> AVD '$AVD_NAME' (720x1280 @ 320dpi, 4 GB RAM)"
# Written by hand: avdmanager is deprecated and would pull the new Android CLI.
avd_dir="$HOME/.android/avd/$AVD_NAME.avd"
mkdir -p "$avd_dir"
cat > "$HOME/.android/avd/$AVD_NAME.ini" <<EOF
avd.ini.encoding=UTF-8
path=$avd_dir
path.rel=avd/$AVD_NAME.avd
target=android-30
EOF
cat > "$avd_dir/config.ini" <<EOF
AvdId=$AVD_NAME
avd.ini.displayname=$AVD_NAME
avd.ini.encoding=UTF-8
PlayStore.enabled=false
abi.type=x86_64
hw.cpu.arch=x86_64
hw.cpu.ncore=4
image.sysdir.1=${SYSIMG//;//}/
tag.id=google_apis
tag.display=Google APIs
disk.dataPartition.size=8G
hw.lcd.width=720
hw.lcd.height=1280
hw.lcd.density=320
hw.initialOrientation=portrait
hw.ramSize=4096
hw.keyboard=yes
hw.gpu.enabled=yes
hw.gpu.mode=host
skin.name=720x1280
skin.path=_no_skin
showDeviceFrame=no
EOF

echo "==> scrcpy (latest GitHub release)"
if [ ! -x "$SCRCPY_DIR/scrcpy" ]; then
  url=$(curl -fsSL https://api.github.com/repos/Genymobile/scrcpy/releases/latest \
    | grep -o 'https://[^"]*scrcpy-linux-x86_64-[^"]*\.tar\.gz' | head -1)
  rm -rf "$SCRCPY_DIR" && mkdir -p "$SCRCPY_DIR"
  curl -fL "$url" | tar -xz -C "$SCRCPY_DIR" --strip-components=1
fi
mkdir -p "$HOME/.local/bin"
# Wrapper: run the bundled binary but use the SDK's adb (two adb versions fight over the server).
cat > "$HOME/.local/bin/scrcpy" <<EOF
#!/usr/bin/env bash
export ADB="$SDK/platform-tools/adb"
exec "$SCRCPY_DIR/scrcpy" "\$@"
EOF
chmod +x "$HOME/.local/bin/scrcpy"

echo "==> v4l2loopback on /dev/video10 (persistent)"
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf >/dev/null
echo "options v4l2loopback video_nr=10 card_label=scrcpy exclusive_caps=1" \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf >/dev/null
if [ ! -e /dev/video10 ]; then
  sudo modprobe -r v4l2loopback 2>/dev/null || true
  sudo modprobe v4l2loopback
fi

echo "==> KVM access"
if [ ! -w /dev/kvm ]; then
  sudo usermod -aG kvm "$USER"
  echo "   added $USER to kvm group: log out/in before starting the emulator"
fi

cat <<EOF

Done. Versions:
  adb:     $("$SDK/platform-tools/adb" version | head -1)
  scrcpy:  $("$HOME/.local/bin/scrcpy" --version 2>/dev/null | head -1)
  loopback: $(ls /dev/video10 2>/dev/null || echo "MISSING")

Next (new terminal, or: source ~/.bashrc):
  emulator -avd $AVD_NAME &
  adb install path/to/clash.apk           # confirm game runs and reaches a match
  scrcpy --v4l2-sink=/dev/video10 --no-window --no-audio --max-fps=60
  ffplay /dev/video10                      # should show the game
EOF
