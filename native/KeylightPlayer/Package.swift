// swift-tools-version:5.9
// Native Keylight player: Metal EDR rendering on macOS (Apple Silicon, XDR display).
// Build & run:  swift run -c release KeylightPlayer --calibration
import PackageDescription

let package = Package(
    name: "KeylightPlayer",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "KeylightPlayer", path: "Sources/KeylightPlayer")
    ]
)
