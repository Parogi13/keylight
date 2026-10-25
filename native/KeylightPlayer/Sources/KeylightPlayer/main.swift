// Keylight native player: Metal EDR rendering on the MacBook XDR display.
//
//   swift run -c release KeylightPlayer                        (library: your videos)
//   swift run -c release KeylightPlayer --calibration
//   swift run -c release KeylightPlayer --image still.png --mask still_mask.png
//   swift run -c release KeylightPlayer <job>/native          (video.mp4 + masks.json)
//   whole-frame saturation starts on at 1.1x; --saturation 1 starts with it off
//
// Library: double-click a video to open it; the trash bin moves it to the macOS Trash.
// Keys: Space play/pause · W wipe · E enhanced · O original · A attention · C calibration
//       Up/Down change subject brightness · drag in the video to move the wipe
//       S whole-frame saturation on/off · [ / ] less / more saturation.
import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var library: LibraryWindowController?
    private var players: [PlayerWindowController] = []
    private var saturation: Float?

    func applicationDidFinishLaunching(_ note: Notification) {
        Theme.registerFonts()
        NSApp.appearance = NSAppearance(named: .darkAqua)
        buildMenu()

        var args = Array(CommandLine.arguments.dropFirst())
        func take(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            let v = args[i + 1]; args.removeSubrange(i...i + 1); return v
        }
        if let sat = take("--saturation") {
            guard let v = Float(sat), v >= 1 else { fatal("--saturation needs a number >= 1, e.g. 1.1") }
            saturation = v
        }
        if args.contains("--calibration") {
            openPlayer(.calibration, title: "Keylight · Calibration")
        } else if let img = take("--image") {
            openPlayer(.still(image: URL(fileURLWithPath: img), mask: take("--mask").map { URL(fileURLWithPath: $0) }),
                       title: "Keylight · \(Library.prettify(URL(fileURLWithPath: img).lastPathComponent))")
        } else if let dir = args.first {
            let url = URL(fileURLWithPath: dir)
            openPlayer(.video(nativeDir: url), title: "Keylight · \(Library.prettify(url.deletingLastPathComponent().lastPathComponent))")
        } else {
            showLibrary()
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func showLibrary() {
        if library == nil {
            let lib = LibraryWindowController()
            lib.onOpen = { [weak self] item in
                self?.openPlayer(.video(nativeDir: item.nativeDir), title: "Keylight · \(item.title)")
            }
            lib.onCalibrate = { [weak self] in self?.openPlayer(.calibration, title: "Keylight · Calibration") }
            lib.onDelete = { [weak self] item in try self?.delete(item) }
            lib.onRename = { [weak self] item, title in try self?.rename(item, to: title) }
            library = lib
        }
        library?.show()
    }

    private func openPlayer(_ source: PlayerSource, title: String) {
        do {
            let p = try PlayerWindowController(source: source, title: title, saturation: saturation)
            p.onClose = { [weak self] closed in self?.players.removeAll { $0 === closed } }
            p.onShowLibrary = { [weak self] in self?.showLibrary() }
            players.append(p)
            p.show()
        } catch {
            let alert = NSAlert()
            alert.messageText = "Couldn't open \(title.replacingOccurrences(of: "Keylight · ", with: ""))"
            alert.informativeText = "\(error)"
            alert.runModal()
            if players.isEmpty && library == nil { fatal("Could not load input: \(error)") }
        }
    }

    /// Moves the video's folder to the Trash (recoverable), closing its player first.
    private func delete(_ item: LibraryItem) throws {
        for p in players {
            if case let .video(dir) = p.source, dir.standardizedFileURL == item.nativeDir.standardizedFileURL { p.close() }
        }
        try FileManager.default.trashItem(at: item.folder, resultingItemURL: nil)
    }

    /// Saves the new name and retitles any open player showing that video.
    private func rename(_ item: LibraryItem, to title: String) throws {
        try Library.rename(item, to: title)
        let shown = Library.scan().first { $0.nativeDir == item.nativeDir }?.title ?? title
        for p in players {
            if case let .video(dir) = p.source, dir.standardizedFileURL == item.nativeDir.standardizedFileURL {
                p.window.title = "Keylight · \(shown)"
            }
        }
    }

    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Keylight", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        let fileItem = NSMenuItem()
        main.addItem(fileItem)
        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem(withTitle: "Library", action: #selector(showLibrary), keyEquivalent: "l")
        fileMenu.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        fileItem.submenu = fileMenu
        NSApp.mainMenu = main
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

func fatal(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

// Top-level code runs on the main thread; tell the compiler so it can call MainActor APIs.
MainActor.assumeIsolated {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let delegate = AppDelegate()
    app.delegate = delegate
    app.run()
}
