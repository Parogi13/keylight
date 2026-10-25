// Metal shaders, compiled at runtime with device.makeLibrary(source:) so the
// package needs no Xcode project or metallib build step.
//
// Colour pipeline (all maths in linear light, extended-linear sRGB output):
//   decoded BT.709 R'G'B' (8-bit, from AVFoundation)  --BT.1886 (gamma 2.4)-->  linear
//   BT.709 and sRGB share primaries, so linear BT.709 == linear sRGB: no matrix.
//   In an EDR layer with extendedLinearSRGB, 1.0 == SDR reference white; values
//   above 1.0 use the display's EDR headroom (up to `headroom`).
//   This matches the encoded-HDR path, which maps SDR white to 203 nits = reference.
let keylightShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float gain;        // subject brightness multiple (1 = off)
    float headroom;    // current EDR headroom (1 = SDR only)
    float strength;    // 0..1, drives the shadow lift
    float wipe;        // 0..1 video-space x of the wipe divider; < 0 = no wipe
    int   mode;        // 0 enhanced, 1 original, 2 attention overlay, 3 calibration
    int   hasMask;
    float peakLimit;   // never exceed this multiple of SDR white
    float saturation;  // whole-frame saturation multiple (1 = off)
};

struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut kl_vertex(uint vid [[vertex_id]]) {
    // One oversized triangle covering the viewport.
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

constant float3 LUMA709 = float3(0.2126, 0.7152, 0.0722);

static float3 to_linear(float3 c) { return pow(max(c, 0.0), 2.4); }

static float3 rolloff(float3 rgb, float peak) {
    // Soft shoulder on luminance above 60% of peak; hue preserved.
    float y = dot(rgb, LUMA709);
    float knee = 0.6 * peak;
    if (y <= knee || peak <= knee) return rgb;
    float r = peak - knee;
    float y2 = knee + r * tanh((y - knee) / r);
    return rgb * (y2 / max(y, 1e-5));
}

// Linear sRGB <-> linear Display P3 (both D65, so luminance is unchanged).
constant float3x3 SRGB_TO_P3 = float3x3(float3(0.8225, 0.0332, 0.0171),
                                        float3(0.1774, 0.9669, 0.0724),
                                        float3(0.0000, 0.0000, 0.9108));
constant float3x3 P3_TO_SRGB = float3x3(float3( 1.2249, -0.0420, -0.0197),
                                        float3(-0.2247,  1.0419, -0.0786),
                                        float3( 0.0000,  0.0000,  1.0979));
constant float3 LUMA_P3 = float3(0.2290, 0.6917, 0.0793);

static float3 saturate_frame(float3 lin, float sat) {
    // Push chroma away from luma in P3, so colours may go past sRGB into the XDR
    // panel's gamut. The push stops where any P3 channel would go negative, so
    // nothing clips and hue is preserved. Luminance is unchanged.
    if (sat <= 1.0) return lin;
    float3 p = SRGB_TO_P3 * lin;
    float y = dot(p, LUMA_P3);
    float3 d = p - y;
    float t = sat;
    for (int i = 0; i < 3; i++) {
        if (d[i] < -1e-6) t = min(t, y / -d[i]);
    }
    return P3_TO_SRGB * (y + max(t, 1.0) * d);
}

static float3 subject(float3 lin, constant Uniforms& u) {
    // Shadow-weighted lift (dark subjects gain more), then the EDR brightness multiple.
    float y = dot(lin, LUMA709);
    float g = 1.0 / (1.0 + 0.5 * u.strength);
    float lift = pow(y + 1e-5, g) / (y + 1e-5);
    float3 e = lin * min(lift, 1.0 + 7.0 * u.strength) * u.gain;
    float peak = max(1.0, min(u.headroom, u.peakLimit));
    return rolloff(e, peak);
}

static float3 calibration(float2 uv, float headroom) {
    // Top: gray steps 0..100% SDR. Bottom: patches at 1x, 2x, 3x, 4x SDR white,
    // clipped to the available headroom (a patch that can't be shown brighter won't be).
    if (uv.y < 0.45) {
        float step = floor(uv.x * 12.0) / 11.0;
        return float3(pow(step, 2.4));
    }
    if (uv.y < 0.55) return float3(0.02);
    float k = floor(uv.x * 4.0) + 1.0;
    float inner = fract(uv.x * 4.0);
    if (inner < 0.08 || inner > 0.92 || uv.y > 0.92) return float3(0.02);
    return float3(min(k, headroom));
}

fragment float4 kl_fragment(VOut in [[stage_in]],
                            texture2d<float> frame [[texture(0)]],
                            texture2d<float> mask  [[texture(1)]],
                            constant Uniforms& u   [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    constexpr sampler soft(filter::linear, mip_filter::linear, address::clamp_to_edge);
    if (u.mode == 3) return float4(calibration(in.uv, u.headroom), 1.0);

    float3 lin = to_linear(frame.sample(s, in.uv).rgb);
    float m = u.hasMask ? saturate(mask.sample(s, in.uv).r) : 0.0;

    if (u.mode == 1) return float4(lin, 1.0);
    if (u.mode == 2) {
        // Attention view: background dimmed and desaturated, subject in true colour.
        float y = dot(lin, LUMA709);
        float3 bg = float3(y) * 0.12;
        return float4(mix(bg, lin, m), 1.0);
    }

    // Two-zone blend, same as the Python renderer: the lift uses a tighter core
    // (m^2.5) so the feathered edge shows the original, not a bright cut-out ring.
    float core = m * m * sqrt(m);
    // Spill: like a real key light, a little of the light falls on the DARK ground around
    // the subject (a blurred mip of the mask), so a lit person sits in the scene instead of
    // reading as a cut-out. Gated by brightness: nothing above ~15% of SDR white gets any,
    // so a bright sky behind a silhouette is never lifted (no halo).
    float spill = 0.0;
    if (u.hasMask) {
        float ms = saturate(mask.sample(soft, in.uv, level(3.0)).r);
        spill = 0.35 * ms * (1.0 - smoothstep(0.02, 0.15, dot(lin, LUMA709)));
    }
    float3 base = saturate_frame(lin, u.saturation);   // whole frame
    float3 out = mix(base, subject(base, u), max(core, spill));

    if (u.wipe >= 0.0) {
        float dx = in.uv.x - u.wipe;
        if (dx < 0.0) out = lin;                                       // left = original
        if (abs(dx) < 0.0012) out = float3(0.96, 0.62, 0.10);          // Tuscan Sun divider
    }
    return float4(out, 1.0);
}
"""
