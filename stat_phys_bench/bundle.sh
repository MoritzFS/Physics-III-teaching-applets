#!/bin/bash
# Builds a double-clickable "Stat Phys Bench.app" next to this folder.
set -e
cd "$(dirname "$0")"
cargo build --release
APP="../Stat Phys Bench.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp target/release/stat_phys_bench "$APP/Contents/MacOS/stat_phys_bench"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Stat Phys Bench</string>
    <key>CFBundleDisplayName</key><string>Stat Phys Bench</string>
    <key>CFBundleIdentifier</key><string>local.teaching.stat-phys-bench</string>
    <key>CFBundleExecutable</key><string>stat_phys_bench</string>
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
