// One Keylight player window: Metal EDR view + transport bar + status line.
// Opened from the library (double-click) or directly from the command line.
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

enum PlayerSource {
    case calibration
    case still(image: URL, mask: URL?)
    case video(nativeDir: URL)
}

@MainActor
final class PlayerWindowController: NSObject, NSWindowDelegate {
    let window: NSWindow
    let source: PlayerSource
    var onClose: ((PlayerWindowController) -> Void)?
    var onShowLibrary: (() -> Void)?

    private let view: KLView
    private let renderer: Renderer
    private let gainSlider = NSSlider(value: 2.2, minValue: 1, maxValue: 4, target: nil, action: nil)
    private let gainLabel = Theme.label("", size: 14)
    private let modeControl = NSSegmentedControl(labels: ["Keylight", "Original", "Wipe", "Attention", "Calibration"],
                                                 trackingMode: .selectOne, target: nil, action: nil)
    private let stats = Theme.label("", size: 13, color: Theme.mist)
    private let playButton = PillButton("Pause", style: .quiet, size: 14, target: nil, action: nil)

    /// `saturation`: nil keeps the renderer default (off); a value above 1 starts it on at that amount.
    init(source: PlayerSource, title: String, saturation: Float? = nil) throws {
        self.source = source
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = title
        window.backgroundColor = Theme.night
        window.isReleasedWhenClosed = false

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
        renderer = try Renderer(view: view)
        super.init()
        view.delegate = renderer
        window.delegate = self

        if let v = saturation {
            renderer.saturationAmount = min(max(v, 1), 2)
            renderer.saturationOn = v > 1
        }
        switch source {
        case .calibration:
            renderer.mode = .calibration
            modeControl.selectedSegment = 4
        case let .still(image, mask):
            try renderer.loadStill(image: image, mask: mask)
        case let .video(dir):
            let masks = dir.appendingPathComponent("masks.json")
            try renderer.loadVideo(url: dir.appendingPathComponent("video.mp4"),
                                   masksJSON: FileManager.default.fileExists(atPath: masks.path) ? masks : nil)
        }
        if modeControl.selectedSegment < 0 { modeControl.selectedSegment = 0 }
        buildUI()
        window.center()
    }

    func show() {
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
    }

    func close() { window.close() }

    func windowWillClose(_ notification: Notification) {
        // Stop sound and drawing now; the window may outlive this controller briefly.
        renderer.stop()
        view.isPaused = true
        view.delegate = nil
        onClose?(self)
    }

    private func buildUI() {
        let root = NSView()
        window.contentView = root
        view.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(view)

        gainSlider.target = self; gainSlider.action = #selector(gainChanged)
        gainSlider.trackFillColor = Theme.key
        modeControl.target = self; modeControl.action = #selector(modeChanged)
        modeControl.selectedSegmentBezelColor = Theme.key
        modeControl.font = Theme.serif(13, .medium)
        playButton.target = self; playButton.action = #selector(togglePlay)
        gainChanged()

        let library = PillButton("Library", style: .link, size: 14, target: self, action: #selector(showLibrary))
        let title = Theme.label("Subject brightness", size: 14)
        let bar = NSStackView(views: [library, playButton, modeControl, title, gainSlider, gainLabel])
        bar.orientation = .horizontal
        bar.spacing = 12
        bar.alignment = .centerY
        bar.edgeInsets = NSEdgeInsets(top: 8, left: 16, bottom: 4, right: 16)
        let column = NSStackView(views: [bar, stats])
        column.orientation = .vertical
        column.alignment = .leading
        column.edgeInsets = NSEdgeInsets(top: 0, left: 0, bottom: 8, right: 0)
        column.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(column)
        gainSlider.widthAnchor.constraint(equalToConstant: 220).isActive = true
        stats.leadingAnchor.constraint(equalTo: column.leadingAnchor, constant: 16).isActive = true

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

    @objc private func showLibrary() { onShowLibrary?() }

    @objc private func gainChanged() {
        renderer.uniforms.gain = Float(gainSlider.doubleValue)
        gainLabel.stringValue = String(format: "%.1fx", gainSlider.doubleValue)
    }

    @objc private func modeChanged() {
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

    @objc private func togglePlay() {
        guard let p = renderer.player else { return }
        if p.rate == 0 { p.play(); playButton.text = "Pause" } else { p.pause(); playButton.text = "Play" }
    }

    private func dragWipe(to point: CGPoint) {
        let scale = view.window?.backingScaleFactor ?? 2
        let r = renderer.videoRect(in: view.drawableSize)
        let x = (point.x * scale - r.minX) / max(r.width, 1)
        renderer.uniforms.wipe = Float(min(max(x, 0), 1))
        renderer.wipeOn = true
        renderer.mode = .enhanced
        modeControl.selectedSegment = 2
    }

    private func handleKey(_ e: NSEvent) -> Bool {
        switch e.keyCode {
        case 49: togglePlay(); return true                         // space
        case 126: gainSlider.doubleValue += 0.2; gainChanged(); return true   // up
        case 125: gainSlider.doubleValue -= 0.2; gainChanged(); return true   // down
        default: break
        }
        switch e.charactersIgnoringModifiers {
        case "s", "S": renderer.saturationOn.toggle(); return true
        case "[", "]":
            let step: Float = e.charactersIgnoringModifiers == "]" ? 0.05 : -0.05
            renderer.saturationAmount = min(max(renderer.saturationAmount + step, 1.05), 2)
            renderer.saturationOn = true
            return true
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
}
