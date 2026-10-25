// Main menu: the library of finished videos, styled like the web app's "Your videos".
// A video is any folder with a native export (native/video.mp4, from scripts/export_native.py):
//   backend/outputs/<job>/native        web-app jobs (name from job.json)
//   bench/<clip>/<backend>/native       benchmark results (YOLO / SAM 2)
// Double-click a name to open the Keylight player; the trash bin at the right end moves the
// video's folder to the macOS Trash (recoverable), after an inline Delete / Keep confirm.
// The pencil (or right-click > Rename) edits the name in place. Per-video settings (custom
// name, subject brightness) live in native/library.json next to the video, so folders and
// the web app's job IDs never change.
import AppKit
import UniformTypeIdentifiers

struct LibraryItem {
    let title: String
    let detail: String
    let nativeDir: URL
    let folder: URL        // what Delete moves to the Trash
    let modified: Date
}

enum Library {
    /// Repo root: KEYLIGHT_ROOT, else the first folder above the executable that holds backend/pipeline.
    static let root: URL = {
        if let env = ProcessInfo.processInfo.environment["KEYLIGHT_ROOT"] { return URL(fileURLWithPath: env) }
        var u = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
        while u.path != "/" {
            u = u.deletingLastPathComponent()
            if FileManager.default.fileExists(atPath: u.appendingPathComponent("backend/pipeline").path) { return u }
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }()

    static var outputs: URL {
        ProcessInfo.processInfo.environment["KEYLIGHT_OUTPUTS"].map { URL(fileURLWithPath: $0) }
            ?? root.appendingPathComponent("backend/outputs")
    }

    static func scan() -> [LibraryItem] {
        var items: [LibraryItem] = []
        for job in subfolders(outputs) {
            let info = (try? JSONSerialization.jsonObject(with: Data(contentsOf: job.appendingPathComponent("job.json")))) as? [String: Any]
            let name = (info?["name"] as? String) ?? job.lastPathComponent
            let backend = ((info?["settings"] as? [String: Any])?["segmentation_backend"] as? String) ?? "yolo"
            if let item = item(title: prettify(name), tag: tag(for: backend), folder: job) { items.append(item) }
        }
        for clip in subfolders(root.appendingPathComponent("bench")) {
            for backend in subfolders(clip) {
                if let item = item(title: prettify(clip.lastPathComponent), tag: tag(for: backend.lastPathComponent),
                                   folder: backend) { items.append(item) }
            }
        }
        return items.sorted { $0.modified > $1.modified }
    }

    /// Which segmentation made the masks: job.json settings for web jobs, the folder name for bench.
    private static func tag(for backend: String) -> String {
        switch backend.lowercased() {
        case "sam2": return "SAM 2"
        case "yolo": return "YOLO"
        default: return backend
        }
    }

    private static func item(title: String, tag: String?, folder: URL) -> LibraryItem? {
        let native = folder.appendingPathComponent("native")
        let video = native.appendingPathComponent("video.mp4")
        guard FileManager.default.fileExists(atPath: video.path) else { return nil }
        let saved = settings(native)
        let custom = saved["title"] as? String
        let masks = native.appendingPathComponent("masks.json")
        var parts: [String] = tag.map { [$0] } ?? []
        if let meta = (try? JSONSerialization.jsonObject(with: Data(contentsOf: masks))) as? [String: Any],
           let count = meta["count"] as? Double ?? (meta["count"] as? Int).map(Double.init),
           let fps = meta["fps"] as? Double ?? (meta["fps"] as? Int).map(Double.init), fps > 0 {
            let s = Int((count / fps).rounded())
            parts.append(String(format: "%d:%02d", s / 60, s % 60))
        }
        let modified = (try? (FileManager.default.fileExists(atPath: masks.path) ? masks : video)
            .resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? nil
        if let gain = (saved["gain"] as? NSNumber)?.doubleValue {
            parts.append(String(format: "%.1f×", gain))   // this video's saved subject brightness
        }
        return LibraryItem(title: custom ?? title, detail: parts.joined(separator: " · "), nativeDir: native, folder: folder,
                           modified: modified ?? .distantPast)
    }

    /// Stores a display name for the video; an empty name goes back to the default.
    /// A video's saved settings (native/library.json): "title", "gain". Empty if none.
    static func settings(_ nativeDir: URL) -> [String: Any] {
        ((try? JSONSerialization.jsonObject(with: Data(contentsOf: nativeDir.appendingPathComponent("library.json"))))
            as? [String: Any]) ?? [:]
    }

    /// Changes some of a video's settings, keeping the others (a rename keeps the
    /// brightness and vice versa). Removes the file when nothing is left.
    static func updateSettings(_ nativeDir: URL, _ change: (inout [String: Any]) -> Void) throws {
        var s = settings(nativeDir)
        change(&s)
        let file = nativeDir.appendingPathComponent("library.json")
        if s.isEmpty {
            if FileManager.default.fileExists(atPath: file.path) { try FileManager.default.removeItem(at: file) }
            return
        }
        try JSONSerialization.data(withJSONObject: s, options: [.prettyPrinted, .sortedKeys]).write(to: file, options: .atomic)
    }

    /// Stores a display name for the video; an empty name goes back to the default.
    static func rename(_ item: LibraryItem, to title: String) throws {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        try updateSettings(item.nativeDir) { s in
            if t.isEmpty { s.removeValue(forKey: "title") } else { s["title"] = t }
        }
    }

    private static func subfolders(_ dir: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                                      options: [.skipsHiddenFiles])) ?? [])
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
    }

    /// Same as the web app: drop the extension, dashes/underscores to spaces, capitalise.
    static func prettify(_ name: String) -> String {
        var base = (name as NSString).deletingPathExtension
        base = base.replacingOccurrences(of: "[-_]+", with: " ", options: .regularExpression)
        return base.prefix(1).uppercased() + base.dropFirst()
    }
}

/// A flipped container so stacks lay out top-down.
final class FlippedView: NSView {
    override var isFlipped: Bool { true }
}

@MainActor
final class LibraryWindowController: NSObject, NSWindowDelegate {
    let window: NSWindow
    var onOpen: ((LibraryItem) -> Void)?
    var onCalibrate: (() -> Void)?
    var onDelete: ((LibraryItem) throws -> Void)?
    var onRename: ((LibraryItem, String) throws -> Void)?

    let processor = Processor()
    private let list = NSStackView()
    private let scroll = NSScrollView()
    private var processingRows: [UUID: ProcessingRow] = [:]
    private let precise = NSButton(checkboxWithTitle: "", target: nil, action: nil)
    private static let preciseKey = "KeylightPreciseMasks"
    private let empty = NSTextField(wrappingLabelWithString: "")

    override init() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 640),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        super.init()
        window.title = "Keylight"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.backgroundColor = Theme.night
        window.minSize = NSSize(width: 520, height: 420)
        window.isReleasedWhenClosed = false
        window.delegate = self
        build()
        window.center()
        processor.onListChange = { [weak self] in self?.reload() }
        processor.onProgress = { [weak self] task in self?.processingRows[task.id]?.update(task) }
        processor.onFinished = { [weak self] in self?.reload() }
    }

    private func build() {
        let root = DropView()
        root.onDrop = { [weak self] urls in self?.addVideos(urls) }
        window.contentView = root

        // Top bar: lamp + glowing wordmark, calibration on the right.
        let mark = LampMark()
        mark.translatesAutoresizingMaskIntoConstraints = false
        mark.widthAnchor.constraint(equalToConstant: 34).isActive = true
        mark.heightAnchor.constraint(equalToConstant: 54).isActive = true
        let brand = NSStackView(views: [mark, Theme.wordmark()])
        brand.spacing = 12
        let calibrate = PillButton("Calibrate display", style: .quiet, size: 14, target: self, action: #selector(calibrateClicked))
        let add = PillButton("Add video", style: .key, size: 15, target: self, action: #selector(addClicked))
        let bar = NSStackView(views: [brand, NSView(), calibrate, add])
        bar.spacing = 12
        bar.distribution = .fill
        bar.alignment = .centerY

        let title = Theme.label("Your videos", size: 20, weight: .semibold)
        precise.attributedTitle = NSAttributedString(string: "Precise masks (SAM 2, several minutes per clip)",
                                                     attributes: [.font: Theme.serif(14), .foregroundColor: Theme.mist])
        precise.state = UserDefaults.standard.bool(forKey: Self.preciseKey) ? .on : .off
        precise.target = self
        precise.action = #selector(preciseChanged)
        let heading = NSStackView(views: [title, NSView(), precise])
        heading.alignment = .firstBaseline
        let hint = Theme.label("Add a video or drop one here. Double-click to play; right-click or the pencil to rename.",
                               size: 15, color: Theme.mist)

        list.orientation = .vertical
        list.alignment = .leading
        list.spacing = 8
        list.translatesAutoresizingMaskIntoConstraints = false
        empty.font = Theme.serif(16)
        empty.textColor = Theme.mist
        empty.stringValue = "No videos yet. Click Add video, or drop a clip onto this window."

        let doc = FlippedView()
        doc.translatesAutoresizingMaskIntoConstraints = false
        doc.addSubview(list)
        scroll.drawsBackground = false
        // The list sits below the transparent title bar; don't let AppKit inset it for one.
        scroll.automaticallyAdjustsContentInsets = false
        scroll.contentInsets = NSEdgeInsetsZero
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.documentView = doc

        for v in [bar, heading, hint, scroll] as [NSView] {
            v.translatesAutoresizingMaskIntoConstraints = false
            root.addSubview(v)
        }
        let g: CGFloat = 32
        NSLayoutConstraint.activate([
            bar.topAnchor.constraint(equalTo: root.topAnchor, constant: 60),
            bar.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: g),
            bar.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -g),
            heading.topAnchor.constraint(equalTo: bar.bottomAnchor, constant: 36),
            heading.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: g),
            heading.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -g),
            hint.topAnchor.constraint(equalTo: heading.bottomAnchor, constant: 4),
            hint.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: g),
            scroll.topAnchor.constraint(equalTo: hint.bottomAnchor, constant: 18),
            scroll.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: g),
            scroll.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -g),
            scroll.bottomAnchor.constraint(equalTo: root.bottomAnchor, constant: -g),
            doc.leadingAnchor.constraint(equalTo: scroll.contentView.leadingAnchor),
            doc.trailingAnchor.constraint(equalTo: scroll.contentView.trailingAnchor),
            doc.topAnchor.constraint(equalTo: scroll.contentView.topAnchor),
            list.topAnchor.constraint(equalTo: doc.topAnchor),
            list.leadingAnchor.constraint(equalTo: doc.leadingAnchor),
            list.trailingAnchor.constraint(equalTo: doc.trailingAnchor),
            list.bottomAnchor.constraint(equalTo: doc.bottomAnchor),
        ])
    }

    func show() {
        reload()
        window.makeKeyAndOrderFront(nil)
    }

    func reload() {
        list.arrangedSubviews.forEach { $0.removeFromSuperview() }
        processingRows = [:]
        for task in processor.tasks {                     // videos being processed come first
            let row = ProcessingRow(task: task)
            row.onCancel = { [weak self] in self?.processor.cancel($0) }
            processingRows[task.id] = row
            list.addArrangedSubview(row)
            row.widthAnchor.constraint(equalTo: list.widthAnchor).isActive = true
        }
        let items = Library.scan()
        if items.isEmpty && processor.tasks.isEmpty {
            list.addArrangedSubview(empty)
            empty.widthAnchor.constraint(equalTo: list.widthAnchor).isActive = true
            return
        }
        for item in items {
            let row = LibraryRow(item: item)
            row.onOpen = { [weak self] in self?.onOpen?($0) }
            // Reload after the row's own event handling has finished, not from inside it.
            row.onDelete = { [weak self] in try self?.onDelete?($0); DispatchQueue.main.async { self?.reload() } }
            row.onRename = { [weak self] in try self?.onRename?($0, $1); DispatchQueue.main.async { self?.reload() } }
            list.addArrangedSubview(row)
            row.widthAnchor.constraint(equalTo: list.widthAnchor).isActive = true
        }
        scrollToTop()
    }

    private func scrollToTop() {
        scroll.layoutSubtreeIfNeeded()
        scroll.contentView.scroll(to: .zero)
        scroll.reflectScrolledClipView(scroll.contentView)
    }

    // New exports show up when you come back to the window.
    func windowDidBecomeKey(_ notification: Notification) { reload() }

    @objc private func calibrateClicked() { onCalibrate?() }

    @objc private func addClicked() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = videoExtensions.compactMap { UTType(filenameExtension: $0) }
        panel.message = "Choose videos to enhance"
        panel.beginSheetModal(for: window) { [weak self] r in
            if r == .OK { self?.addVideos(panel.urls) }
        }
    }

    private func addVideos(_ urls: [URL]) {
        let videos = urls.filter { videoExtensions.contains($0.pathExtension.lowercased()) }
        guard !videos.isEmpty, let focus = askFocus(count: videos.count) else { return }
        processor.add(videos, precise: precise.state == .on, focus: focus)
    }

    /// Optional focus instruction for Claude, once per batch; nil if the user cancels.
    private func askFocus(count: Int) -> String? {
        let alert = NSAlert()
        alert.messageText = "What should Keylight follow?"
        alert.informativeText = "Optional, for example \"the cyclist in red\". If Claude is set up, it picks the "
            + "matching subject while the video is processed. Leave it empty to let Keylight choose"
            + (count > 1 ? " (applies to all \(count) videos)." : ".")
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 320, height: 24))
        field.placeholderString = "Leave empty to let Keylight choose"
        alert.accessoryView = field
        alert.addButton(withTitle: "Enhance")
        alert.addButton(withTitle: "Cancel")
        alert.window.initialFirstResponder = field
        guard alert.runModal() == .alertFirstButtonReturn else { return nil }
        return field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @objc private func preciseChanged() {
        UserDefaults.standard.set(precise.state == .on, forKey: Self.preciseKey)
    }
}

/// One video: a name pill (double-click opens) and a trash bin at the right end.
@MainActor
final class LibraryRow: NSView, NSTextFieldDelegate {
    let item: LibraryItem
    var onOpen: ((LibraryItem) -> Void)?
    var onDelete: ((LibraryItem) throws -> Void)?
    var onRename: ((LibraryItem, String) throws -> Void)?

    private let pill: NamePill
    private let pencil = IconButton(symbol: "pencil", hover: Theme.key)
    private let trash = IconButton(symbol: "trash", hover: Theme.ember)
    private let confirm = NSStackView()
    private let error = Theme.label("", size: 14, color: Theme.ember)

    init(item: LibraryItem) {
        self.item = item
        pill = NamePill(title: item.title, detail: item.detail)
        super.init(frame: .zero)
        pill.onDoubleClick = { [weak self] in
            guard let self = self else { return }
            self.onOpen?(self.item)
        }
        trash.target = self
        trash.action = #selector(askDelete)
        trash.toolTip = "Delete"
        trash.setAccessibilityLabel("Delete \(item.title)")
        pencil.target = self
        pencil.action = #selector(beginRename)
        pencil.toolTip = "Rename"
        pencil.setAccessibilityLabel("Rename \(item.title)")
        pill.name.delegate = self
        let menu = NSMenu()
        menu.addItem(withTitle: "Open", action: #selector(openFromMenu), keyEquivalent: "").target = self
        menu.addItem(withTitle: "Rename…", action: #selector(beginRename), keyEquivalent: "").target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Delete…", action: #selector(askDelete), keyEquivalent: "").target = self
        pill.menu = menu

        let del = PillButton("Delete", style: .danger, size: 15, target: self, action: #selector(doDelete))
        let keep = PillButton("Keep", style: .link, size: 15, target: self, action: #selector(cancelDelete))
        confirm.setViews([del, keep], in: .leading)
        confirm.spacing = 12
        confirm.isHidden = true
        error.isHidden = true

        let line = NSStackView(views: [pill, pencil, trash, confirm])
        line.spacing = 8
        line.alignment = .centerY
        let column = NSStackView(views: [line, error])
        column.orientation = .vertical
        column.alignment = .leading
        column.spacing = 4
        column.translatesAutoresizingMaskIntoConstraints = false
        addSubview(column)
        NSLayoutConstraint.activate([
            column.topAnchor.constraint(equalTo: topAnchor),
            column.bottomAnchor.constraint(equalTo: bottomAnchor),
            column.leadingAnchor.constraint(equalTo: leadingAnchor),
            column.trailingAnchor.constraint(equalTo: trailingAnchor),
            line.widthAnchor.constraint(equalTo: column.widthAnchor),
        ])
        pill.setContentHuggingPriority(.defaultLow, for: .horizontal)
    }

    required init?(coder: NSCoder) { fatalError() }

    @objc private func openFromMenu() { onOpen?(item) }

    @objc private func askDelete() {
        trash.isHidden = true
        pencil.isHidden = true
        confirm.isHidden = false
        error.isHidden = true
    }

    @objc private func cancelDelete() {
        confirm.isHidden = true
        trash.isHidden = false
        pencil.isHidden = false
    }

    // MARK: rename in place (Return saves, Esc cancels, empty restores the default name)

    @objc private func beginRename() {
        cancelDelete()
        error.isHidden = true
        pill.setEditing(true)
        window?.makeFirstResponder(pill.name)
        pill.name.currentEditor()?.selectAll(nil)
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy selector: Selector) -> Bool {
        if selector == #selector(NSResponder.cancelOperation(_:)) {
            pill.name.abortEditing()          // discards the edit without ending it as a commit
            pill.name.stringValue = item.title
            pill.setEditing(false)
            return true
        }
        return false
    }

    func controlTextDidEndEditing(_ obj: Notification) {
        pill.setEditing(false)
        let new = pill.name.stringValue
        guard new != item.title else { return }
        do {
            try onRename?(item, new)
        } catch {
            pill.name.stringValue = item.title
            self.error.stringValue = "Couldn't rename: \(error.localizedDescription)"
            self.error.isHidden = false
        }
    }

    @objc private func doDelete() {
        do {
            try onDelete?(item)
        } catch {
            self.error.stringValue = "Couldn't delete: \(error.localizedDescription)"
            self.error.isHidden = false
            cancelDelete()
        }
    }
}

/// The name "tab": Graphite pill, Tuscan Sun border on hover, double-click to open.
final class NamePill: NSView {
    var onDoubleClick: (() -> Void)?
    let name: NSTextField
    private var editing = false
    private var hovering = false { didSet { updateBorder() } }

    init(title: String, detail: String) {
        name = Theme.label(title, size: 17)
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = Theme.dusk.cgColor
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        layer?.borderColor = Theme.rim.cgColor
        setAccessibilityRole(.button)
        setAccessibilityLabel("Open \(title)")

        name.lineBreakMode = .byTruncatingTail
        name.focusRingType = .none
        name.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        let info = Theme.label(detail, size: 14, color: Theme.mist)
        let row = NSStackView(views: [name, NSView(), info])
        row.alignment = .centerY
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)
        NSLayoutConstraint.activate([
            heightAnchor.constraint(equalToConstant: 44),
            row.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            row.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            row.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }

    /// Turns the name into an edit field (lit border, raised fill) and back.
    func setEditing(_ on: Bool) {
        editing = on
        name.isEditable = on
        name.isSelectable = on
        name.drawsBackground = on
        name.backgroundColor = Theme.dusk2
        layer?.backgroundColor = (on ? Theme.dusk2 : Theme.dusk).cgColor
        updateBorder()
    }

    private func updateBorder() { layer?.borderColor = (hovering || editing ? Theme.key : Theme.rim).cgColor }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect], owner: self))
    }
    override func mouseEntered(with event: NSEvent) { hovering = true }
    override func mouseExited(with event: NSEvent) { hovering = false }
    override func mouseDown(with event: NSEvent) { if event.clickCount == 2 && !editing { onDoubleClick?() } }
    override func accessibilityPerformPress() -> Bool { onDoubleClick?(); return true }
}

/// Small icon square (44x44, outlined like web .library-delete): trash turns Ember on hover,
/// the rename pencil turns Tuscan Sun.
final class IconButton: NSButton {
    private let hoverColor: NSColor
    private var hovering = false { didSet { restyle() } }

    init(symbol: String, hover: NSColor) {
        hoverColor = hover
        super.init(frame: .zero)
        isBordered = false
        title = ""
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        let cfg = NSImage.SymbolConfiguration(pointSize: 15, weight: .regular)
        image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)?.withSymbolConfiguration(cfg)
        imagePosition = .imageOnly
        translatesAutoresizingMaskIntoConstraints = false
        widthAnchor.constraint(equalToConstant: 44).isActive = true
        heightAnchor.constraint(equalToConstant: 44).isActive = true
        restyle()
    }

    required init?(coder: NSCoder) { fatalError() }

    // The default bezel style pads the frame beyond the alignment rect, so the drawn box
    // would be taller than the row and offset; the layer is the whole button here.
    override var alignmentRectInsets: NSEdgeInsets { NSEdgeInsetsZero }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect], owner: self))
    }
    override func mouseEntered(with event: NSEvent) { hovering = true }
    override func mouseExited(with event: NSEvent) { hovering = false }

    private func restyle() {
        contentTintColor = hovering ? hoverColor : Theme.mist
        layer?.borderColor = (hovering ? hoverColor : Theme.rim).cgColor
    }
}
