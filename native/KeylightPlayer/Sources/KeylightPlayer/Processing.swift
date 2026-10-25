// Adding videos from the library: each file runs scripts/process_native.py (the web app's
// pipeline + the native export) in the experiment's Python, one at a time (they share the
// GPU). Progress arrives as "PROGRESS {json}" lines; the finished job lands in
// backend/outputs/<id>/, so it also appears in the web app.
import AppKit

/// Same set as the web app's VIDEO_EXT (backend/main.py).
let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "mkv", "webm", "avi"]

@MainActor
final class Processor {
    struct Task {
        let id = UUID()
        let source: URL
        let precise: Bool          // SAM 2 instead of YOLO
        let focus: String          // optional instruction for Claude ("" = Keylight chooses)
        var label = "Waiting"
        var progress = 0.0
        var error: String?
    }

    private(set) var tasks: [Task] = []
    private var running: (id: UUID, process: Process)?
    private var finished = false         // the running task reported DONE or ERROR
    private var buffer = Data()
    /// Something visible changed: a task was added, removed or failed.
    var onListChange: (() -> Void)?
    /// Progress of a running task moved.
    var onProgress: ((Task) -> Void)?
    /// A video finished and is in the library now.
    var onFinished: (() -> Void)?

    static var python: URL? {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["KEYLIGHT_PYTHON"] { return URL(fileURLWithPath: env) }
        return [".venv-sam2/bin/python", ".venv/bin/python"].map { Library.root.appendingPathComponent($0) }
            .first { fm.isExecutableFile(atPath: $0.path) }
    }

    func add(_ urls: [URL], precise: Bool, focus: String) {
        for url in urls where videoExtensions.contains(url.pathExtension.lowercased()) {
            tasks.append(Task(source: url, precise: precise, focus: focus))
        }
        onListChange?()
        startNext()
    }

    /// Cancels a running or waiting video, or dismisses a failed one.
    func cancel(_ id: UUID) {
        if let r = running, r.id == id {
            r.process.terminate()       // SIGTERM: the script removes its half-built job folder
            return                      // removed from the list when it exits
        }
        tasks.removeAll { $0.id == id }
        onListChange?()
    }

    /// App is quitting: stop the running job (it cleans up after itself) and drop the queue.
    func stopAll() {
        running?.process.terminate()
        tasks.removeAll { $0.id != running?.id }
    }

    private func startNext() {
        guard running == nil, let task = tasks.first(where: { $0.error == nil }) else { return }
        guard let python = Self.python else {
            fail(task.id, "No Python environment found. Run ./scripts/setup_sam2.sh, or set KEYLIGHT_PYTHON.")
            return startNext()
        }
        let p = Process()
        p.executableURL = python
        p.arguments = [Library.root.appendingPathComponent("scripts/process_native.py").path, task.source.path,
                       "--backend", task.precise ? "sam2" : "yolo", "--focus", task.focus]
        p.currentDirectoryURL = Library.root
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Self.logHandle()
        buffer = Data()
        finished = false
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let chunk = h.availableData
            if chunk.isEmpty { h.readabilityHandler = nil; return }   // EOF: stop polling
            DispatchQueue.main.async { self?.consume(chunk, id: task.id) }
        }
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            DispatchQueue.main.async { self?.exited(task.id, status: status) }
        }
        do {
            try p.run()
            running = (task.id, p)
            update(task.id) { $0.label = "Starting" }
        } catch {
            fail(task.id, "Couldn't start processing: \(error.localizedDescription)")
            startNext()
        }
    }

    private func consume(_ chunk: Data, id: UUID) {
        buffer.append(chunk)
        while let nl = buffer.firstIndex(of: 0x0A) {
            let line = String(decoding: buffer[buffer.startIndex..<nl], as: UTF8.self)
            buffer.removeSubrange(buffer.startIndex...nl)
            guard let space = line.firstIndex(of: " ") else { continue }
            let kind = line[..<space]
            let json = (try? JSONSerialization.jsonObject(with: Data(line[line.index(after: space)...].utf8))) as? [String: Any] ?? [:]
            switch kind {
            case "PROGRESS":
                update(id) {
                    $0.label = json["label"] as? String ?? $0.label
                    $0.progress = (json["progress"] as? Double) ?? $0.progress
                }
            case "DONE":
                finished = true
            case "ERROR":
                finished = true
                let msg = json["message"] as? String ?? "Processing failed"
                if msg != "Cancelled" { fail(id, msg) }
            default: break
            }
        }
    }

    private func exited(_ id: UUID, status: Int32) {
        running = nil
        let failed = tasks.first { $0.id == id }?.error != nil
        if status == 0 {
            tasks.removeAll { $0.id == id }
            onListChange?()
            onFinished?()
        } else if failed {
            onListChange?()               // keep the failed row so the message can be read
        } else if !finished && status != 15 && status != 130 {
            fail(id, "Processing stopped unexpectedly (exit \(status)). Details: \(Self.logURL.path)")
        } else {
            tasks.removeAll { $0.id == id } // cancelled
            onListChange?()
        }
        startNext()
    }

    private func update(_ id: UUID, _ change: (inout Task) -> Void) {
        guard let i = tasks.firstIndex(where: { $0.id == id }) else { return }
        change(&tasks[i])
        onProgress?(tasks[i])
    }

    private func fail(_ id: UUID, _ message: String) {
        guard let i = tasks.firstIndex(where: { $0.id == id }) else { return }
        tasks[i].error = message
        onListChange?()
    }

    /// The pipeline's own output (stderr) goes to a log file, for when something fails.
    static let logURL = FileManager.default.temporaryDirectory.appendingPathComponent("keylight-process.log")
    private static func logHandle() -> FileHandle {
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        return (try? FileHandle(forWritingTo: logURL)) ?? FileHandle.nullDevice
    }
}

/// A video being processed: name, stage, percent and a lit progress meter (web .meter).
final class ProcessingRow: NSView {
    let taskID: UUID
    var onCancel: ((UUID) -> Void)?
    private let status = Theme.label("", size: 14, color: Theme.mist)
    private let fill = NSView()
    private var fillWidth: NSLayoutConstraint!
    private var progress = 0.0
    private let track = NSView()
    private let cancel = IconButton(symbol: "xmark", hover: Theme.ember)

    init(task: Processor.Task) {
        taskID = task.id
        super.init(frame: .zero)
        let pill = NSView()
        pill.wantsLayer = true
        pill.layer?.backgroundColor = Theme.dusk.cgColor
        pill.layer?.cornerRadius = 8
        pill.layer?.borderWidth = 1
        pill.layer?.borderColor = Theme.rim.cgColor

        let extra = (task.precise ? "  ·  SAM 2" : "") + (task.focus.isEmpty ? "" : "  ·  “\(task.focus)”")
        let name = Theme.label(Library.prettify(task.source.lastPathComponent) + extra, size: 17)
        name.lineBreakMode = .byTruncatingTail
        name.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        status.alignment = .right
        let top = NSStackView(views: [name, NSView(), status])
        top.alignment = .centerY

        track.wantsLayer = true
        track.layer?.backgroundColor = Theme.dusk2.cgColor
        track.layer?.cornerRadius = 2
        fill.wantsLayer = true
        fill.layer?.backgroundColor = Theme.key.cgColor
        fill.layer?.cornerRadius = 2
        fill.layer?.shadowColor = Theme.key.cgColor
        fill.layer?.shadowOpacity = 0.6
        fill.layer?.shadowRadius = 6
        fill.layer?.shadowOffset = .zero
        track.layer?.masksToBounds = false
        for v in [top, track, fill] as [NSView] { v.translatesAutoresizingMaskIntoConstraints = false; pill.addSubview(v) }
        fillWidth = fill.widthAnchor.constraint(equalToConstant: 0)

        cancel.target = self
        cancel.action = #selector(cancelClicked)
        let line = NSStackView(views: [pill, cancel])
        line.spacing = 8
        line.alignment = .centerY
        line.translatesAutoresizingMaskIntoConstraints = false
        addSubview(line)
        NSLayoutConstraint.activate([
            line.topAnchor.constraint(equalTo: topAnchor), line.bottomAnchor.constraint(equalTo: bottomAnchor),
            line.leadingAnchor.constraint(equalTo: leadingAnchor), line.trailingAnchor.constraint(equalTo: trailingAnchor),
            pill.heightAnchor.constraint(equalToConstant: 58),
            top.topAnchor.constraint(equalTo: pill.topAnchor, constant: 10),
            top.leadingAnchor.constraint(equalTo: pill.leadingAnchor, constant: 16),
            top.trailingAnchor.constraint(equalTo: pill.trailingAnchor, constant: -16),
            track.leadingAnchor.constraint(equalTo: pill.leadingAnchor, constant: 16),
            track.trailingAnchor.constraint(equalTo: pill.trailingAnchor, constant: -16),
            track.bottomAnchor.constraint(equalTo: pill.bottomAnchor, constant: -10),
            track.heightAnchor.constraint(equalToConstant: 4),
            fill.leadingAnchor.constraint(equalTo: track.leadingAnchor),
            fill.topAnchor.constraint(equalTo: track.topAnchor),
            fill.bottomAnchor.constraint(equalTo: track.bottomAnchor),
            fillWidth,
        ])
        update(task)
    }

    required init?(coder: NSCoder) { fatalError() }

    func update(_ task: Processor.Task) {
        if let e = task.error {
            status.stringValue = e
            status.textColor = Theme.ember
            status.toolTip = e
            cancel.toolTip = "Dismiss"
            track.isHidden = true
            fill.isHidden = true
            return
        }
        status.textColor = Theme.mist
        status.stringValue = task.label == "Waiting" ? "Waiting" : "\(task.label) · \(Int((task.progress * 100).rounded()))%"
        cancel.toolTip = "Cancel"
        progress = min(max(task.progress, 0), 1)
        needsLayout = true
    }

    override func layout() {
        super.layout()
        fillWidth.constant = track.bounds.width * CGFloat(progress)   // follows window resizes too
    }

    @objc private func cancelClicked() { onCancel?(taskID) }
}

/// Library content view that accepts video files dropped from Finder (lit border while dragging).
final class DropView: NSView {
    var onDrop: (([URL]) -> Void)?
    private let ring = CALayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = Theme.night.cgColor
        registerForDraggedTypes([.fileURL])
        ring.borderColor = Theme.key.cgColor
        ring.borderWidth = 2
        ring.cornerRadius = 18
        ring.isHidden = true
        layer?.addSublayer(ring)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layout() {
        super.layout()
        ring.frame = bounds.insetBy(dx: 10, dy: 10)
    }

    private func videos(_ info: NSDraggingInfo) -> [URL] {
        let urls = info.draggingPasteboard.readObjects(forClasses: [NSURL.self],
                                                        options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        return urls.filter { videoExtensions.contains($0.pathExtension.lowercased()) }
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard !videos(sender).isEmpty else { return [] }
        ring.isHidden = false
        return .copy
    }
    override func draggingExited(_ sender: NSDraggingInfo?) { ring.isHidden = true }
    override func draggingEnded(_ sender: NSDraggingInfo) { ring.isHidden = true }
    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = videos(sender)
        ring.isHidden = true
        guard !urls.isEmpty else { return false }
        onDrop?(urls)
        return true
    }
}
