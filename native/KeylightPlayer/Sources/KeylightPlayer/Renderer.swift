import AppKit
import AVFoundation
import CoreVideo
import Metal
import MetalKit
import QuartzCore

/// Layout must match `struct Uniforms` in the shader (8 x 4 bytes).
struct Uniforms {
    var gain: Float = 2.2
    var headroom: Float = 1
    var strength: Float = 0.6
    var wipe: Float = -1
    var mode: Int32 = 0
    var hasMask: Int32 = 0
    var peakLimit: Float = 4
    var pad: Float = 0
}

enum ViewMode: Int32 { case enhanced = 0, original = 1, attention = 2, calibration = 3 }

/// Per-frame attention masks exported by `scripts/export_native.py`:
/// masks.json {width, height, count, fps} + masks.bin (count * width * height uint8).
final class MaskStore {
    let width: Int, height: Int, count: Int, fps: Double
    private let data: Data

    init(jsonURL: URL) throws {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL)) as! [String: Any]
        width = meta["width"] as! Int
        height = meta["height"] as! Int
        count = meta["count"] as! Int
        fps = (meta["fps"] as? Double) ?? Double(meta["fps"] as! Int)
        let bin = jsonURL.deletingLastPathComponent().appendingPathComponent("masks.bin")
        data = try Data(contentsOf: bin, options: .alwaysMapped)  // memory-mapped, not loaded
        precondition(data.count >= width * height * count, "masks.bin is shorter than masks.json says")
    }

    /// Frame index for a presentation time: frame k is shown at k / fps (CFR video).
    func index(forSeconds s: Double) -> Int { max(0, min(count - 1, Int(s * fps + 1e-3))) }

    func upload(_ i: Int, to texture: MTLTexture) {
        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!.advanced(by: i * width * height)
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                            withBytes: base, bytesPerRow: width)
        }
    }
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache?
    var uniforms = Uniforms()
    var mode: ViewMode = .enhanced
    var wipeOn = false

    // Sources
    private(set) var frameTexture: MTLTexture?
    private var cvTexture: CVMetalTexture?          // keeps the video frame alive while in use
    private var maskTexture: MTLTexture?
    private var masks: MaskStore?
    private var lastMaskIndex = -1
    var player: AVPlayer?
    private var videoOutput: AVPlayerItemVideoOutput?
    private(set) var videoSize = CGSize(width: 16, height: 9)

    // Stats (shown in the HUD)
    private(set) var gpuMs: Double = 0
    private(set) var frameIndex = 0
    var onStats: ((String) -> Void)?
    private var lastHUD = CACurrentMediaTime()

    init(view: MTKView) throws {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice() else { throw KLError("No Metal device") }
        self.device = device
        guard let q = device.makeCommandQueue() else { throw KLError("No command queue") }
        queue = q
        let library = try device.makeLibrary(source: keylightShaderSource, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = library.makeFunction(name: "kl_vertex")
        desc.fragmentFunction = library.makeFunction(name: "kl_fragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipeline = try device.makeRenderPipelineState(descriptor: desc)
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        textureCache = cache
        super.init()
        // 1x1 black mask so the shader always has something bound.
        maskTexture = makeMaskTexture(width: 1, height: 1)
    }

    private func makeMaskTexture(width: Int, height: Int) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        d.usage = .shaderRead
        d.storageMode = .shared
        return device.makeTexture(descriptor: d)
    }

    // MARK: sources

    func loadStill(image: URL, mask: URL?) throws {
        let loader = MTKTextureLoader(device: device)
        let opts: [MTKTextureLoader.Option: Any] = [.SRGB: false,
                                                    .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue)]
        let tex = try loader.newTexture(URL: image, options: opts)
        frameTexture = tex
        videoSize = CGSize(width: tex.width, height: tex.height)
        if let mask = mask {
            maskTexture = try loader.newTexture(URL: mask, options: opts)
            uniforms.hasMask = 1
        }
    }

    func loadVideo(url: URL, masksJSON: URL?) throws {
        if let masksJSON = masksJSON {
            let store = try MaskStore(jsonURL: masksJSON)
            masks = store
            maskTexture = makeMaskTexture(width: store.width, height: store.height)
            uniforms.hasMask = 1
        }
        let item = AVPlayerItem(url: url)
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        let out = AVPlayerItemVideoOutput(pixelBufferAttributes: attrs)
        item.add(out)
        videoOutput = out
        let p = AVPlayer(playerItem: item)
        p.actionAtItemEnd = .none
        NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main) { _ in
            p.seek(to: .zero)
            p.play()
        }
        player = p   // AVPlayer is the clock; audio plays natively and stays in sync.
        p.play()
    }

    /// Pull the frame AVPlayer wants on screen right now (if new) and its mask.
    private func updateVideoFrame() {
        guard let out = videoOutput, let cache = textureCache else { return }
        let t = out.itemTime(forHostTime: CACurrentMediaTime())
        guard out.hasNewPixelBuffer(forItemTime: t),
              let pb = out.copyPixelBuffer(forItemTime: t, itemTimeForDisplay: nil) else { return }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pb, nil, .bgra8Unorm, w, h, 0, &cvTex)
        if let cvTex = cvTex, let tex = CVMetalTextureGetTexture(cvTex) {
            cvTexture = cvTex
            frameTexture = tex
            videoSize = CGSize(width: w, height: h)
        }
        if let masks = masks, let mtex = maskTexture {
            let i = masks.index(forSeconds: CMTimeGetSeconds(t))
            frameIndex = i
            if i != lastMaskIndex {
                masks.upload(i, to: mtex)
                lastMaskIndex = i
            }
        }
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    /// Aspect-fit rectangle of the video inside the drawable, in pixels.
    func videoRect(in drawable: CGSize) -> CGRect {
        let a = videoSize.width / max(videoSize.height, 1)
        var w = drawable.width, h = w / a
        if h > drawable.height { h = drawable.height; w = h * a }
        return CGRect(x: (drawable.width - w) / 2, y: (drawable.height - h) / 2, width: w, height: h)
    }

    func draw(in view: MTKView) {
        updateVideoFrame()
        // Current EDR headroom: changes with the brightness slider and ambient light.
        let screen = view.window?.screen ?? NSScreen.main
        uniforms.headroom = Float(screen?.maximumExtendedDynamicRangeColorComponentValue ?? 1)
        uniforms.mode = mode.rawValue
        if !wipeOn { uniforms.wipe = -1 }

        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        if mode == .calibration || frameTexture != nil {
            let r = mode == .calibration ? CGRect(origin: .zero, size: view.drawableSize) : videoRect(in: view.drawableSize)
            enc.setViewport(MTLViewport(originX: Double(r.minX), originY: Double(r.minY),
                                        width: Double(r.width), height: Double(r.height), znear: 0, zfar: 1))
            enc.setRenderPipelineState(pipeline)
            enc.setFragmentTexture(frameTexture ?? maskTexture, index: 0)
            enc.setFragmentTexture(maskTexture, index: 1)
            var u = uniforms
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        enc.endEncoding()
        let keep = cvTexture  // hold the CoreVideo texture until the GPU is done with it
        cmd.addCompletedHandler { [weak self] cb in
            _ = keep
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            DispatchQueue.main.async { self?.gpuMs = 0.9 * (self?.gpuMs ?? 0) + 0.1 * ms }
        }
        cmd.present(drawable)
        cmd.commit()

        let now = CACurrentMediaTime()
        if now - lastHUD > 0.25 {
            lastHUD = now
            let potential = screen?.maximumPotentialExtendedDynamicRangeColorComponentValue ?? 1
            let peak = min(Double(uniforms.headroom), Double(uniforms.peakLimit))
            onStats?(String(format: "EDR headroom %.2fx (potential %.1fx)  ·  subject peak %.2fx SDR white  ·  GPU %.2f ms  ·  frame %ld",
                            uniforms.headroom, potential, max(1, peak), gpuMs, frameIndex))
        }
    }
}

struct KLError: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}
