#!/bin/bash
# Builds a double-clickable "Optics Bench.app" next to this folder.
set -e
cd "$(dirname "$0")"
cargo build --release
APP="../Optics Bench.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp target/release/optics_bench "$APP/Contents/MacOS/optics_bench"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Optics Bench</string>
    <key>CFBundleDisplayName</key><string>Optics Bench</string>
    <key>CFBundleIdentifier</key><string>local.teaching.optics-bench</string>
    <key>CFBundleExecutable</key><string>optics_bench</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --deep -s - "$APP" 2>/dev/null || true
echo "built $APP"
