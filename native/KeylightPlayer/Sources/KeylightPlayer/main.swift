// Keylight native player: Metal EDR rendering on the MacBook XDR display.
//
//   swift run -c release KeylightPlayer --calibration
//   swift run -c release KeylightPlayer --image still.png --mask still_mask.png
//   swift run -c release KeylightPlayer <job>/native          (video.mp4 + masks.json)
//
// Keys: Space play/pause · W wipe · E enhanced · O original · A attention · C calibration
//       Up/Down change subject brightness · drag in the video to move the wipe.
import AppKit
import MetalKit

@MainActor
final class KLView: MTKView {
    var onKey: ((NSEvent) -> Bool)?
    var onDrag: ((CGPoint) -> Void)?
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with e: NSEvent) { if !(onKey?(e) ?? false) { super.keyDown(with: e) } }
    override func mouseDown(with e: NSEvent) { onDrag?(convert(e.locationInWindow, from: nil)) }
    override func mouseDragged(with e: NSEvent) { onDrag?(convert(e.locationInWindow, from: nil)) }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: KLView!
    var renderer: Renderer!
    let gainSlider = NSSlider(value: 2.2, minValue: 1, maxValue: 4, target: nil, action: nil)
    let gainLabel = NSTextField(labelWithString: "")
    let modeControl = NSSegmentedControl(labels: ["Keylight", "Original", "Wipe", "Attention", "Calibration"],
                                         trackingMode: .selectOne, target: nil, action: nil)
    let stats = NSTextField(labelWithString: "")
    let playButton = NSButton(title: "Pause", target: nil, action: nil)

    func applicationDidFinishLaunching(_ note: Notification) {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "Keylight (native EDR)"
        window.backgroundColor = NSColor(calibratedRed: 0.141, green: 0.141, blue: 0.137, alpha: 1)

        view = KLView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        // --- EDR configuration -------------------------------------------------
        // Float16 drawable + extended-linear colour space + EDR opt-in: values above
        // 1.0 are shown brighter than SDR white, up to the screen's current headroom.
        view.colorPixelFormat = .rgba16Float
        view.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
        if let layer = view.layer as? CAMetalLayer {
            layer.wantsExtendedDynamicRangeContent = true
            layer.pixelFormat = .rgba16Float
            layer.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
        }
        view.preferredFramesPerSecond = 60
        view.framebufferOnly = true

        do {
            renderer = try Renderer(view: view)
        } catch {
            fatal("Metal setup failed: \(error)")
        }
        view.delegate = renderer
        configureSource()
        buildUI()
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
    }

    func configureSource() {
        var args = Array(CommandLine.arguments.dropFirst())
        func take(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            let v = args[i + 1]; args.removeSubrange(i...i + 1); return v
        }
        do {
            if args.contains("--calibration") {
                renderer.mode = .calibration
                modeControl.selectedSegment = 4
            } else if let img = take("--image") {
                try renderer.loadStill(image: URL(fileURLWithPath: img), mask: take("--mask").map { URL(fileURLWithPath: $0) })
            } else if let dir = args.first {
                let base = URL(fileURLWithPath: dir)
                let video = base.appendingPathComponent("video.mp4")
                let masks = base.appendingPathComponent("masks.json")
                try renderer.loadVideo(url: video, masksJSON: FileManager.default.fileExists(atPath: masks.path) ? masks : nil)
            } else {
                renderer.mode = .calibration
                modeControl.selectedSegment = 4
            }
        } catch {
            fatal("Could not load input: \(error)")
        }
        if modeControl.selectedSegment < 0 { modeControl.selectedSegment = 0 }
    }

    func buildUI() {
        let root = NSView()
        window.contentView = root
        view.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(view)

        gainSlider.target = self; gainSlider.action = #selector(gainChanged)
        modeControl.target = self; modeControl.action = #selector(modeChanged)
        playButton.target = self; playButton.action = #selector(togglePlay)
        playButton.bezelStyle = .rounded
        stats.textColor = NSColor(calibratedRed: 0.81, green: 0.86, blue: 0.84, alpha: 1)
        stats.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        gainLabel.textColor = .white
        gainChanged()

        let title = NSTextField(labelWithString: "Subject brightness")
        title.textColor = .white
        let bar = NSStackView(views: [playButton, modeControl, title, gainSlider, gainLabel])
        bar.orientation = .horizontal
        bar.spacing = 12
        bar.edgeInsets = NSEdgeInsets(top: 8, left: 12, bottom: 4, right: 12)
        let column = NSStackView(views: [bar, stats])
        column.orientation = .vertical
        column.alignment = .leading
        column.edgeInsets = NSEdgeInsets(top: 0, left: 0, bottom: 8, right: 0)
        column.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(column)
        gainSlider.widthAnchor.constraint(equalToConstant: 220).isActive = true
        stats.leadingAnchor.constraint(equalTo: column.leadingAnchor, constant: 12).isActive = true

        NSLayoutConstraint.activate([
            view.topAnchor.constraint(equalTo: root.topAnchor),
            view.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            view.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            view.bottomAnchor.constraint(equalTo: column.topAnchor),
            column.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            column.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            column.bottomAnchor.constraint(equalTo: root.bottomAnchor),
        ])

        renderer.onStats = { [weak self] s in self?.stats.stringValue = s }
        view.onKey = { [weak self] e in self?.handleKey(e) ?? false }
        view.onDrag = { [weak self] p in self?.dragWipe(to: p) }
        playButton.isHidden = renderer.player == nil
    }

    @objc func gainChanged() {
        renderer?.uniforms.gain = Float(gainSlider.doubleValue)
        gainLabel.stringValue = String(format: "%.1fx", gainSlider.doubleValue)
    }

    @objc func modeChanged() {
        switch modeControl.selectedSegment {
        case 0: renderer.mode = .enhanced; renderer.wipeOn = false
        case 1: renderer.mode = .original; renderer.wipeOn = false
        case 2:
            renderer.mode = .enhanced; renderer.wipeOn = true
            if renderer.uniforms.wipe < 0 { renderer.uniforms.wipe = 0.5 }
        case 3: renderer.mode = .attention; renderer.wipeOn = false
        default: renderer.mode = .calibration; renderer.wipeOn = false
        }
    }

    @objc func togglePlay() {
        guard let p = renderer.player else { return }
        if p.rate == 0 { p.play(); playButton.title = "Pause" } else { p.pause(); playButton.title = "Play" }
    }

    func dragWipe(to point: CGPoint) {
        let scale = view.window?.backingScaleFactor ?? 2
        let r = renderer.videoRect(in: view.drawableSize)
        let x = (point.x * scale - r.minX) / max(r.width, 1)
        renderer.uniforms.wipe = Float(min(max(x, 0), 1))
        renderer.wipeOn = true
        renderer.mode = .enhanced
        modeControl.selectedSegment = 2
    }

    func handleKey(_ e: NSEvent) -> Bool {
        switch e.keyCode {
        case 49: togglePlay(); return true                         // space
        case 126: gainSlider.doubleValue += 0.2; gainChanged(); return true   // up
        case 125: gainSlider.doubleValue -= 0.2; gainChanged(); return true   // down
        default: break
        }
        switch e.charactersIgnoringModifiers?.lowercased() {
        case "e": modeControl.selectedSegment = 0
        case "o": modeControl.selectedSegment = 1
        case "w": modeControl.selectedSegment = 2
        case "a": modeControl.selectedSegment = 3
        case "c": modeControl.selectedSegment = 4
        default: return false
        }
        modeChanged()
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

func fatal(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
