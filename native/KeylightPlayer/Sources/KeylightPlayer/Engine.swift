// Client for the local Keylight engine (the Python backend started by ./scripts/start.sh,
// on http://127.0.0.1:8000 on this Mac). "Add video" in the library hands it a file path;
// the engine runs the same pipeline as the web app (detection, optional Claude subject
// choice, temporal attention) and writes <job>/native/ when it finishes, which the library
// then lists. The video itself never leaves this computer.
import Foundation

@MainActor
enum Engine {
    static let base = URL(string: ProcessInfo.processInfo.environment["KEYLIGHT_ENGINE"] ?? "http://127.0.0.1:8000")!

    struct Status {
        let state: String       // queued | processing | complete | error
        let label: String       // user-facing stage name
        let progress: Double    // 0...1
        let error: String?
    }

    /// Starts a job for a local file; returns its job ID.
    static func submit(_ file: URL, focus: String) async throws -> String {
        var req = URLRequest(url: base.appendingPathComponent("api/process_path"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "input_path": file.path,
            "settings": ["focus": focus],
        ])
        let (data, resp) = try await URLSession.shared.data(for: req)
        let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        guard (resp as? HTTPURLResponse)?.statusCode == 200, let id = obj?["id"] as? String else {
            throw KLError((obj?["detail"] as? String) ?? "The engine couldn't start this video.")
        }
        return id
    }

    static func status(_ id: String) async throws -> Status {
        let (data, _) = try await URLSession.shared.data(from: base.appendingPathComponent("api/status/\(id)"))
        guard let o = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw KLError("Lost track of the job. Is the engine still running?")
        }
        return Status(state: o["status"] as? String ?? "",
                      label: o["label"] as? String ?? "",
                      progress: (o["progress"] as? NSNumber)?.doubleValue ?? 0,
                      error: o["error"] as? String)
    }
}
