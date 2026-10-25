// Keylight look, ported from the web app (frontend/src/index.css, components/Logo.tsx):
// Carbon Black page, Graphite surfaces, Soft Linen type, one light source (Tuscan Sun)
// reserved for the logo and the few things that are "lit". Type is IBM Plex Serif,
// bundled in Fonts/ (SIL OFL, see Fonts/OFL.txt) so it works offline.
import AppKit
import CoreText

extension NSColor {
    convenience init(hex: UInt32, alpha: CGFloat = 1) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xff) / 255, green: CGFloat((hex >> 8) & 0xff) / 255,
                  blue: CGFloat(hex & 0xff) / 255, alpha: alpha)
    }
}

enum Theme {
    static let night = NSColor(hex: 0x242423)    // Carbon Black: page
    static let dusk = NSColor(hex: 0x333533)     // Graphite: panels
    static let dusk2 = NSColor(hex: 0x3c3e3c)    // raised / hover
    static let rim = NSColor(hex: 0x4a4d4a)      // borders, tracks
    static let mist = NSColor(hex: 0xcfdbd5)     // secondary text
    static let paper = NSColor(hex: 0xe8eddf)    // primary text
    static let key = NSColor(hex: 0xf5cb5c)      // Tuscan Sun: the light
    static let keyInk = NSColor(hex: 0x242423)
    static let ember = NSColor(hex: 0xff9b7a)    // destructive
    static let muted = NSColor(hex: 0x8f9690)

    static func registerFonts() {
        guard let dir = Bundle.module.url(forResource: "Fonts", withExtension: nil),
              let urls = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
        for url in urls where url.pathExtension == "ttf" {
            var error: Unmanaged<CFError>?
            if !CTFontManagerRegisterFontsForURL(url as CFURL, .process, &error) {
                FileHandle.standardError.write("font \(url.lastPathComponent) not registered\n".data(using: .utf8)!)
            }
        }
    }

    /// IBM Plex Serif at a weight; Georgia if the bundled font is missing (as the web CSS falls back).
    static func serif(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
        let name: String
        switch weight {
        case .medium: name = "IBMPlexSerif-Medium"
        case .semibold: name = "IBMPlexSerif-SemiBold"
        case .bold, .heavy, .black: name = "IBMPlexSerif-Bold"
        default: name = "IBMPlexSerif-Regular"
        }
        return NSFont(name: name, size: size) ?? NSFont(name: "Georgia", size: size) ?? .systemFont(ofSize: size, weight: weight)
    }

    static func label(_ text: String, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor = paper) -> NSTextField {
        let l = NSTextField(labelWithString: text)
        l.font = serif(size, weight)
        l.textColor = color
        return l
    }

    /// The glowing "keylight" wordmark: the name is the light.
    static func wordmark(size: CGFloat = 32) -> NSTextField {
        let l = label("keylight", size: size, weight: .medium, color: key)
        let glow = NSShadow()
        glow.shadowColor = key.withAlphaComponent(0.6)
        glow.shadowBlurRadius = size * 0.45
        glow.shadowOffset = .zero
        l.shadow = glow
        return l
    }
}

/// Keylight mark: a gooseneck key light whose panel is lit in Tuscan Sun.
/// Same geometry as the web Logo.tsx SVG (viewBox 40 40 260 430).
final class LampMark: NSView {
    override var isFlipped: Bool { true }   // SVG coordinates: y grows downwards

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let vb = CGRect(x: 40, y: 40, width: 260, height: 430)
        let s = min(bounds.width / vb.width, bounds.height / vb.height)
        ctx.saveGState()
        ctx.translateBy(x: (bounds.width - vb.width * s) / 2, y: (bounds.height - vb.height * s) / 2)
        ctx.scaleBy(x: s, y: s)
        ctx.translateBy(x: -vb.minX, y: -vb.minY)

        let paper = Theme.paper.cgColor
        ctx.setFillColor(paper)
        ctx.fillEllipse(in: CGRect(x: 88 - 62, y: 452 - 11, width: 124, height: 22))          // base
        ctx.addPath(CGPath(roundedRect: CGRect(x: 84, y: 208, width: 8, height: 244), cornerWidth: 3, cornerHeight: 3, transform: nil))
        ctx.addPath(CGPath(roundedRect: CGRect(x: 80, y: 232, width: 16, height: 22), cornerWidth: 4, cornerHeight: 4, transform: nil))
        ctx.fillPath()                                                                         // stand + clamp
        let arm = CGMutablePath()
        arm.move(to: CGPoint(x: 88, y: 210))
        arm.addCurve(to: CGPoint(x: 150, y: 128), control1: CGPoint(x: 88, y: 168), control2: CGPoint(x: 108, y: 142))
        ctx.addPath(arm)
        ctx.setStrokeColor(paper)
        ctx.setLineWidth(8)
        ctx.setLineCap(.round)
        ctx.strokePath()                                                                       // gooseneck

        ctx.translateBy(x: 225, y: 112)                                                        // rotate(-28 225 112)
        ctx.rotate(by: -28 * .pi / 180)
        ctx.translateBy(x: -225, y: -112)
        let r = CGRect(x: 160, y: 52, width: 130, height: 120)
        let panel = CGPath(roundedRect: r, cornerWidth: 38, cornerHeight: 38, transform: nil)
        ctx.saveGState()
        ctx.addPath(panel)
        ctx.clip()
        let colors = [NSColor(hex: 0xFFF6D6), NSColor(hex: 0xF8DB86), NSColor(hex: 0xF5CB5C)].map(\.cgColor) as CFArray
        if let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colors, locations: [0, 0.55, 1]) {
            let c = CGPoint(x: r.minX + 0.45 * r.width, y: r.minY + 0.40 * r.height)            // cx 45%, cy 40%
            let radius = 0.7 * hypot(r.width, r.height) / 2.0.squareRoot()                     // r 70% (SVG objectBoundingBox)
            ctx.drawRadialGradient(g, startCenter: c, startRadius: 0, endCenter: c, endRadius: radius, options: [.drawsAfterEndLocation])
        }
        ctx.restoreGState()
        ctx.addPath(panel)
        ctx.setStrokeColor(paper)
        ctx.setLineWidth(9)
        ctx.strokePath()                                                                       // lit panel
        ctx.restoreGState()
    }
}

/// A rounded, layer-drawn button in the web app's styles (.btn-key, .btn-quiet, .btn-danger, .link).
final class PillButton: NSButton {
    enum Style { case key, quiet, danger, link }
    private let style: Style
    private var hovering = false { didSet { restyle() } }
    /// The label; set this (not `title`) so the styling is kept.
    var text: String { didSet { restyle(); invalidateIntrinsicContentSize() } }

    init(_ title: String, style: Style, size: CGFloat = 15, target: AnyObject?, action: Selector?) {
        self.style = style
        text = title
        super.init(frame: .zero)
        self.target = target
        self.action = action
        isBordered = false
        wantsLayer = true
        font = Theme.serif(size, style == .link ? .regular : .bold)
        focusRingType = .exterior
        restyle()
    }

    required init?(coder: NSCoder) { fatalError() }

    // Layer-drawn: the frame must be exactly the alignment rect (no bezel padding).
    override var alignmentRectInsets: NSEdgeInsets { NSEdgeInsetsZero }

    override var intrinsicContentSize: NSSize {
        let s = super.intrinsicContentSize
        return style == .link ? s : NSSize(width: s.width + (style == .danger ? 28 : 40), height: max(s.height + 14, 36))
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect], owner: self))
    }
    override func mouseEntered(with event: NSEvent) { hovering = true }
    override func mouseExited(with event: NSEvent) { hovering = false }

    private func restyle() {
        guard let layer = layer else { return }
        let fg: NSColor
        switch style {
        case .key:
            layer.backgroundColor = (hovering ? Theme.key.blended(withFraction: 0.08, of: .white) ?? Theme.key : Theme.key).cgColor
            layer.cornerRadius = 18; layer.borderWidth = 0; fg = Theme.keyInk
            layer.shadowColor = Theme.key.cgColor; layer.shadowOpacity = 0.35; layer.shadowRadius = 12; layer.shadowOffset = .zero
        case .quiet:
            layer.backgroundColor = Theme.dusk2.cgColor; layer.cornerRadius = 18
            layer.borderWidth = 1; layer.borderColor = (hovering ? Theme.mist : Theme.rim).cgColor; fg = Theme.paper
        case .danger:
            layer.backgroundColor = Theme.ember.cgColor; layer.cornerRadius = 8; layer.borderWidth = 0; fg = Theme.keyInk
        case .link:
            layer.backgroundColor = NSColor.clear.cgColor; layer.borderWidth = 0; fg = hovering ? Theme.paper : Theme.mist
        }
        var attrs: [NSAttributedString.Key: Any] = [.font: font ?? Theme.serif(15), .foregroundColor: fg]
        if style == .link { attrs[.underlineStyle] = NSUnderlineStyle.single.rawValue }
        attributedTitle = NSAttributedString(string: text, attributes: attrs)
    }
}
