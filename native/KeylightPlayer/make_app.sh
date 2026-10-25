#!/usr/bin/env bash
# Build Keylight.app: a double-clickable Mac app (Spotlight, Launchpad, Dock) for the
# native player and its library.
#
#   ./make_app.sh                 -> ~/Applications/Keylight.app
#   ./make_app.sh /some/folder    -> /some/folder/Keylight.app
#
# Re-run after pulling or changing the Swift code. The app finds this repo through
# KEYLIGHT_ROOT (set in its Info.plist), so keep the repo where it is when you build;
# PATH includes Homebrew so "Add video" can find ffmpeg when launched from Finder.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
DEST_DIR="${1:-$HOME/Applications}"
APP="$DEST_DIR/Keylight.app"
TMP="$DEST_DIR/.Keylight.app.building"

swift build -c release
BIN="$(swift build -c release --show-bin-path)"

rm -rf "$TMP"
mkdir -p "$TMP/Contents/MacOS" "$TMP/Contents/Resources"
cp "$BIN/KeylightPlayer" "$TMP/Contents/MacOS/KeylightPlayer"
cp -R "$BIN/KeylightPlayer_KeylightPlayer.bundle/Fonts" "$TMP/Contents/Resources/Fonts"   # IBM Plex Serif (OFL)
cp AppIcon.icns "$TMP/Contents/Resources/AppIcon.icns"

cat > "$TMP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Keylight</string>
  <key>CFBundleDisplayName</key><string>Keylight</string>
  <key>CFBundleIdentifier</key><string>dev.keylight.player</string>
  <key>CFBundleExecutable</key><string>KeylightPlayer</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSEnvironment</key>
  <dict>
    <key>KEYLIGHT_ROOT</key><string>${ROOT}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
PLIST

codesign --force --sign - "$TMP" >/dev/null 2>&1   # ad-hoc: a normal local app to macOS
rm -rf "$APP"
mv "$TMP" "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"
echo "Built $APP (repo: $ROOT)"
