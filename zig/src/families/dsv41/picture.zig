//! A picture for the vision tower, as DeepSeek's MIT reference (image_processor.py) and the served lane make it, with
//! DeepSeek Recipe's decode and grid: the image decoded to 8-bit RGB as Recipe's OpenCV preprocessing gives it (PNG
//! here: every colour type, bit depth and interlace; 16-bit samples divided by 257, transparency blended onto #FDFDFD),
//! its grid (Recipe's calc_resize: plan_image_grid, the patch grid of the ViT and the token grid after the 3x3 aligner,
//! repeated on its own result until it holds), ImageOps.pad to the grid's pixel size (PIL's 8-bit BICUBIC resample of
//! the image fitted inside, 127 grey around it), then each value (v / 255 - 0.5) / 0.5 rounded to bf16, in patch order
//! [patch][channel][row][column] (the ViT's linear patch embed reads each patch flattened that way). Every step is
//! integer or IEEE float math in the reference's order, so for an opaque 8-bit image whose grid holds at the first
//! pass the patches are the lane's byte for byte (gate: tf-dsv41-picture against the recorded `decode` points).
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The vision settings the grid depends on (config.json vision_config; DeepSeek-V4.1-Flash's values by default).
pub const Config = struct {
    patch: usize = 14,
    ratio: usize = 3, // the aligner's downsample
    max_tokens: usize = 1024, // an image span's tokens at most
    min_pixels: usize = 544 * 544,
};

/// Six little-endian u32 grid dimensions, padded to align the following bf16 patches to 16 bytes.
pub const header_bytes = 32;

pub const Prepared = struct { tokens: u32, payload: []align(16) const u8 };

/// Packed RGB rows, w * h * 3 bytes.
pub const Rgb = struct {
    w: usize,
    h: usize,
    px: []u8,

    pub fn deinit(r: *Rgb, a: Allocator) void {
        a.free(r.px);
        r.* = undefined;
    }
};

/// plan_image_grid's result: the padded pixel size (a whole number of patches each way), the ViT's patch grid and the
/// LLM's token grid.
pub const Grid = struct {
    best_h: usize,
    best_w: usize,
    n_vit_h: usize,
    n_vit_w: usize,
    n_llm_h: usize,
    n_llm_w: usize,

    /// num_image_tokens: [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END].
    pub fn spanTokens(g: Grid) usize {
        return g.n_llm_h * (g.n_llm_w + 1) + 2;
    }

    pub fn patches(g: Grid) usize {
        return g.n_vit_h * g.n_vit_w;
    }
};

/// The token types of an image span (image_token_types): IMAGE_START, then per row n_llm_w IMAGE and an
/// IMAGE_NEW_LINE, then IMAGE_END.
pub const Type = enum(u8) { start = 0, image = 1, newline = 2, end = 3 };

pub fn spanTypes(g: Grid, out: []Type) void {
    std.debug.assert(out.len == g.spanTokens());
    var i: usize = 0;
    out[i] = .start;
    i += 1;
    for (0..g.n_llm_h) |_| {
        for (0..g.n_llm_w) |_| {
            out[i] = .image;
            i += 1;
        }
        out[i] = .newline;
        i += 1;
    }
    out[i] = .end;
}

extern "c" fn sqrt(x: f64) f64;

/// Python's round() of a float: half to even.
fn pyRound(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    var r = f;
    if (d > 0.5) r = f + 1 else if (d == 0.5) r = if (@mod(f, 2.0) == 0) f else f + 1;
    return @intFromFloat(r);
}

fn ceilDiv(a: f64, b: f64) usize {
    return @intFromFloat(@ceil(a / b));
}

/// llm_grid: the token grid the aligner makes of a patch grid of this pixel size.
fn llmGrid(best_h: usize, best_w: usize, p: usize, r: usize) [2]usize {
    const fh: f64 = @floatFromInt(best_h / p);
    const fw: f64 = @floatFromInt(best_w / p);
    const fr: f64 = @floatFromInt(r);
    return .{ @intFromFloat(@ceil(fh / fr)), @intFromFloat(@ceil(fw / fr)) };
}

fn spanOf(n_llm_h: usize, n_llm_w: usize) usize {
    return n_llm_h * (n_llm_w + 1) + 2;
}

/// solve_resize_ratio: the largest aspect-preserving pixel size whose token grid fits in max_tokens.
fn solveResize(height: f64, width: f64, p: usize, r: usize, max_n: usize) [2]usize {
    const ratio = height / width;
    const max_w = sqrt(@as(f64, @floatFromInt(max_n - 2)) / ratio + 0.25) - 0.5;
    const max_h = max_w * ratio;
    const cell = p * r;
    if (max_w < 1.0) return .{ (max_n - 2) / 2 * cell, cell };
    if (max_h < 1.0) return .{ cell, (max_n - 3) * cell };
    const fcell: f64 = @floatFromInt(cell);
    const beta = @min(@floor(max_w) * fcell / width, @floor(max_h) * fcell / height);
    const fp: f64 = @floatFromInt(p);
    return .{ @as(usize, @intFromFloat(@floor(height * beta / fp))) * p, @as(usize, @intFromFloat(@floor(width * beta / fp))) * p };
}

/// calc_resize for an image of width x height pixels: planOnce repeated on its own pixel size until the grid holds, at
/// most 10 passes; null when it does not hold by then.
pub fn plan(width0: usize, height0: usize, cfg: Config) ?Grid {
    var g = planOnce(width0, height0, cfg);
    for (1..10) |_| {
        const next = planOnce(g.best_w, g.best_h, cfg);
        if (std.meta.eql(next, g)) return g;
        g = next;
    }
    return null;
}

/// plan_image_grid for an image of width x height pixels (max_wh_ratio unset, as this model's config has it).
fn planOnce(width0: usize, height0: usize, cfg: Config) Grid {
    const p = cfg.patch;
    var width: f64 = @floatFromInt(width0);
    var height: f64 = @floatFromInt(height0);
    const area = width0 * height0;
    if (area > 0 and area < cfg.min_pixels) {
        // ratio = sqrt(min_pixels / (width * height)); width = int(width * ratio); height = int(height * ratio)
        const ratio = sqrt(@as(f64, @floatFromInt(cfg.min_pixels)) / @as(f64, @floatFromInt(area)));
        width = @trunc(width * ratio);
        height = @trunc(height * ratio);
    }
    const fp: f64 = @floatFromInt(p);
    var best_w = ceilDiv(width, fp) * p;
    var best_h = ceilDiv(height, fp) * p;
    var g = llmGrid(best_h, best_w, p, cfg.ratio);
    if (spanOf(g[0], g[1]) > cfg.max_tokens) {
        const hw = solveResize(height, width, p, cfg.ratio, cfg.max_tokens);
        best_h = hw[0];
        best_w = hw[1];
        g = llmGrid(best_h, best_w, p, cfg.ratio);
    }
    return .{ .best_h = best_h, .best_w = best_w, .n_vit_h = best_h / p, .n_vit_w = best_w / p, .n_llm_h = g[0], .n_llm_w = g[1] };
}

// ---------------------------------------------------------------------------------------------------- PNG

pub const Error = error{ NotPng, BadPng, UnsupportedPng, TooLarge, OutOfMemory };

const signature = "\x89PNG\r\n\x1a\n";

fn be32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .big);
}

/// The pixel count the lane's PIL decodes at most before it refuses (DecompressionBombError past twice its
/// Image.MAX_IMAGE_PIXELS: 134,217,728 in the served lane, its probes 2026-10-09).
pub const max_pixels: usize = 134217728;

/// The pixels an image's header declares (PNG's IHDR), or null.
pub fn pixels(data: []const u8) ?usize {
    if (data.len < 24 or !std.mem.eql(u8, data[0..8], "\x89PNG\r\n\x1a\n")) return null;
    return @as(usize, std.mem.readInt(u32, data[16..20], .big)) * std.mem.readInt(u32, data[20..24], .big);
}

/// The background transparency is blended onto: #FDFDFD.
const background: u32 = 0xfd;

/// A colour sample of opacity `alpha` over the background: (c * alpha + 253 * (255 - alpha) + 127) / 255.
pub fn blend(c: u8, alpha: u8) u8 {
    return @intCast((@as(u32, c) * alpha + background * (255 - alpha) + 127) / 255);
}

/// A 16-bit sample to 8 bits: v / 257 rounded to nearest (never a tie).
pub fn from16(v: u16) u8 {
    return @intCast((@as(u32, v) + 128) / 257);
}

/// PNG bytes to 8-bit RGB as DeepSeek Recipe's OpenCV decode gives them: 16-bit samples divided by 257, grey of 1, 2
/// and 4 bits scaled to 0..255 (0 / 255, x 85, x 17), palette indices through PLTE (missing entries black), and alpha
/// blended onto the background: the alpha channel, or tRNS for a palette (entries past it opaque) or an RGB image (the
/// one colour transparent; at 8 bits its low bytes). tRNS of a grey image, of an image with an alpha channel, of the
/// wrong length, longer than the PLTE before it, after IDAT or after another tRNS is ignored. Chunk CRCs are checked.
pub fn decodePng(a: Allocator, data: []const u8) Error!Rgb {
    if (data.len < 8 or !std.mem.eql(u8, data[0..8], signature)) return error.NotPng;
    var at: usize = 8;
    var w: usize = 0;
    var h: usize = 0;
    var depth: u8 = 0;
    var ctype: u8 = 0;
    var interlace: u8 = 0;
    var palette: [256][3]u8 = @splat(.{ 0, 0, 0 });
    var entries: usize = 0;
    var trns: []const u8 = &.{};
    var trns_entries: usize = 0; // the PLTE entries before tRNS
    var have_trns = false;
    var have_ihdr = false;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(a);
    while (true) {
        if (at + 12 > data.len) return error.BadPng;
        const len = be32(data[at..]);
        const kind = data[at + 4 .. at + 8];
        if (len > data.len - at - 12) return error.BadPng;
        const body = data[at + 8 .. at + 8 + len];
        const crc = be32(data[at + 8 + len ..]);
        var hc = std.hash.Crc32.init();
        hc.update(kind);
        hc.update(body);
        if (hc.final() != crc) return error.BadPng;
        at += 12 + len;
        if (std.mem.eql(u8, kind, "IHDR")) {
            if (len != 13) return error.BadPng;
            w = be32(body[0..]);
            h = be32(body[4..]);
            depth = body[8];
            ctype = body[9];
            if (body[10] != 0 or body[11] != 0 or body[12] > 1) return error.UnsupportedPng;
            interlace = body[12];
            have_ihdr = true;
        } else if (std.mem.eql(u8, kind, "PLTE")) {
            if (len % 3 != 0 or len > 768) return error.BadPng;
            for (0..len / 3) |i| palette[i] = .{ body[3 * i], body[3 * i + 1], body[3 * i + 2] };
            entries = len / 3;
        } else if (std.mem.eql(u8, kind, "tRNS")) {
            if (!have_trns and idat.items.len == 0) {
                trns = body;
                trns_entries = entries;
                have_trns = true;
            }
        } else if (std.mem.eql(u8, kind, "IDAT")) {
            try idat.appendSlice(a, body);
        } else if (std.mem.eql(u8, kind, "IEND")) {
            break;
        }
    }
    if (!have_ihdr or w == 0 or h == 0) return error.BadPng;
    if (w * h > max_pixels) return error.TooLarge;
    const channels: usize = switch (ctype) {
        0 => 1,
        2 => 3,
        3 => 1,
        4 => 2,
        6 => 4,
        else => return error.UnsupportedPng,
    };
    const ok_depth = switch (ctype) {
        0 => depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16,
        3 => depth == 1 or depth == 2 or depth == 4 or depth == 8,
        else => depth == 8 or depth == 16,
    };
    if (!ok_depth) return error.UnsupportedPng;
    const bits = channels * depth;
    const bpp = @max(1, bits / 8);

    // the passes' filtered rows, inflated
    var raw: std.Io.Writer.Allocating = .init(a);
    defer raw.deinit();
    {
        var in: std.Io.Reader = .fixed(idat.items);
        var z: std.compress.flate.Decompress = .init(&in, .zlib, &.{});
        _ = z.reader.streamRemaining(&raw.writer) catch return error.BadPng;
    }
    const zdata = raw.written();

    // samples per pixel at full depth for the whole image: [h][w][channels] as u16
    const samples = try a.alloc(u16, w * h * channels);
    defer a.free(samples);
    const passes: [7][4]usize = .{ .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 }, .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 } };
    const one: [1][4]usize = .{.{ 0, 0, 1, 1 }};
    const list: []const [4]usize = if (interlace == 1) &passes else &one;
    var off: usize = 0;
    var prev: std.ArrayList(u8) = .empty;
    defer prev.deinit(a);
    var cur: std.ArrayList(u8) = .empty;
    defer cur.deinit(a);
    for (list) |ps| {
        const x0 = ps[0];
        const y0 = ps[1];
        const dx = ps[2];
        const dy = ps[3];
        if (x0 >= w or y0 >= h) continue;
        const pw = (w - x0 + dx - 1) / dx;
        const ph = (h - y0 + dy - 1) / dy;
        const stride = (pw * bits + 7) / 8;
        try prev.resize(a, stride);
        @memset(prev.items, 0);
        try cur.resize(a, stride);
        for (0..ph) |row| {
            if (off + 1 + stride > zdata.len) return error.BadPng;
            const filter = zdata[off];
            const line = zdata[off + 1 .. off + 1 + stride];
            off += 1 + stride;
            const c = cur.items;
            const p = prev.items;
            for (0..stride) |i| {
                const left: u8 = if (i >= bpp) c[i - bpp] else 0;
                const up = p[i];
                const ul: u8 = if (i >= bpp) p[i - bpp] else 0;
                c[i] = switch (filter) {
                    0 => line[i],
                    1 => line[i] +% left,
                    2 => line[i] +% up,
                    3 => line[i] +% @as(u8, @intCast((@as(u16, left) + up) / 2)),
                    4 => line[i] +% paeth(left, up, ul),
                    else => return error.BadPng,
                };
            }
            // samples of this row's pixels
            const y = y0 + row * dy;
            for (0..pw) |col| {
                const x = x0 + col * dx;
                for (0..channels) |ch| {
                    const k = col * channels + ch;
                    samples[(y * w + x) * channels + ch] = sampleAt(c, k, depth);
                }
            }
            std.mem.swap(std.ArrayList(u8), &prev, &cur);
        }
    }

    var palette_alpha: [256]u8 = @splat(255);
    var key: ?[3]u16 = null;
    switch (ctype) {
        3 => if (trns.len > 0 and trns.len <= trns_entries) @memcpy(palette_alpha[0..trns.len], trns),
        2 => if (trns.len == 6) {
            const mask: u16 = if (depth == 16) 0xffff else 0xff;
            key = .{ std.mem.readInt(u16, trns[0..2], .big) & mask, std.mem.readInt(u16, trns[2..4], .big) & mask, std.mem.readInt(u16, trns[4..6], .big) & mask };
        },
        else => {},
    }
    const px = try a.alloc(u8, w * h * 3);
    errdefer a.free(px);
    for (0..w * h) |i| {
        const s = samples[i * channels ..][0..channels];
        var alpha: u8 = 255;
        const rgb: [3]u8 = switch (ctype) {
            0 => blk: {
                const v = grey(s[0], depth);
                break :blk .{ v, v, v };
            },
            2 => blk: {
                if (key) |k| if (s[0] == k[0] and s[1] == k[1] and s[2] == k[2]) {
                    alpha = 0;
                };
                break :blk .{ sample8(s[0], depth), sample8(s[1], depth), sample8(s[2], depth) };
            },
            6 => blk: {
                alpha = sample8(s[3], depth);
                break :blk .{ sample8(s[0], depth), sample8(s[1], depth), sample8(s[2], depth) };
            },
            4 => blk: {
                alpha = sample8(s[1], depth);
                const v = sample8(s[0], depth);
                break :blk .{ v, v, v };
            },
            3 => blk: {
                alpha = palette_alpha[@min(s[0], 255)];
                break :blk palette[@min(s[0], 255)];
            },
            else => unreachable,
        };
        px[i * 3 ..][0..3].* = if (alpha == 255) rgb else .{ blend(rgb[0], alpha), blend(rgb[1], alpha), blend(rgb[2], alpha) };
    }
    return .{ .w = w, .h = h, .px = px };
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p: i16 = @as(i16, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

/// Sample k of a row at this bit depth (most significant bits first).
fn sampleAt(row: []const u8, k: usize, depth: u8) u16 {
    return switch (depth) {
        16 => (@as(u16, row[2 * k]) << 8) | row[2 * k + 1],
        8 => row[k],
        else => blk: {
            const per = 8 / depth;
            const byte = row[k / per];
            const shift: u3 = @intCast(8 - depth * (k % per + 1));
            const mask: u8 = @intCast((@as(u16, 1) << @intCast(depth)) - 1);
            break :blk (byte >> shift) & mask;
        },
    };
}

/// An 8- or 16-bit sample to 8 bits.
fn sample8(v: u16, depth: u8) u8 {
    return if (depth == 16) from16(v) else @intCast(v);
}

/// A grey sample of any depth to 8 bits: 1 bit (0 / 255), 2 bits (x 85), 4 bits (x 17), 8 and 16 bits as sample8.
fn grey(v: u16, depth: u8) u8 {
    return switch (depth) {
        1 => if (v != 0) 255 else 0,
        2 => @intCast(v * 85),
        4 => @intCast(v * 17),
        else => sample8(v, depth),
    };
}

// ---------------------------------------------------------------------------------------------- low detail

/// The long side DeepSeek Recipe's low detail limits an image to before its grid.
pub const low_detail_side: usize = 512;

/// f32 rounded half to even, clamped to 0..255: OpenCV's saturate_cast<uchar>.
fn saturate8(x: f32) u8 {
    const f = @floor(x);
    const d = x - f;
    var r = f;
    if (d > 0.5 or (d == 0.5 and @mod(f, 2.0) != 0)) r = f + 1;
    return @intFromFloat(std.math.clamp(r, 0, 255));
}

const AreaWeight = struct { di: usize, si: usize, alpha: f32 };

/// OpenCV's computeResizeAreaTab: each destination cell's source pixels and their overlap weights.
fn areaWeights(a: Allocator, ssize: usize, dsize: usize, cn: usize, scale: f64) ![]AreaWeight {
    var tab: std.ArrayList(AreaWeight) = .empty;
    errdefer tab.deinit(a);
    const s: f64 = @floatFromInt(ssize);
    for (0..dsize) |dx| {
        const fsx1 = @as(f64, @floatFromInt(dx)) * scale;
        const fsx2 = fsx1 + scale;
        const cell = @min(scale, s - fsx1);
        var sx1: usize = @intFromFloat(@ceil(fsx1));
        var sx2: usize = @intFromFloat(@floor(fsx2));
        sx2 = @min(sx2, ssize - 1);
        sx1 = @min(sx1, sx2);
        if (@as(f64, @floatFromInt(sx1)) - fsx1 > 1e-3)
            try tab.append(a, .{ .di = dx * cn, .si = (sx1 - 1) * cn, .alpha = @floatCast((@as(f64, @floatFromInt(sx1)) - fsx1) / cell) });
        for (sx1..sx2) |sx| try tab.append(a, .{ .di = dx * cn, .si = sx * cn, .alpha = @floatCast(1.0 / cell) });
        const rest = fsx2 - @as(f64, @floatFromInt(sx2));
        if (rest > 1e-3) try tab.append(a, .{ .di = dx * cn, .si = sx2 * cn, .alpha = @floatCast(@min(@min(rest, 1.0), cell) / cell) });
    }
    return tab.toOwnedSlice(a);
}

/// The image scaled down so its long side is at most `max_side`, as DeepSeek Recipe's limit_image_mat does it: each
/// side times max_side / long side, rounded half away from zero, at least 1, then OpenCV's resize with INTER_AREA in
/// its generic code (x86, no IPP). Whole-number factors average blocks (2 x 2 as (sum + 2) / 4, others as sum times
/// 1 / area in f32); other factors weigh each source pixel by its overlap, accumulated in f32 row by row. Null when
/// the long side is within the limit.
pub fn limitLongSide(a: Allocator, rgb: Rgb, max_side: usize) error{OutOfMemory}!?Rgb {
    const long = @max(rgb.w, rgb.h);
    if (long <= max_side) return null;
    const ratio = @as(f64, @floatFromInt(max_side)) / @as(f64, @floatFromInt(long));
    const dw: usize = @max(1, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(rgb.w)) * ratio))));
    const dh: usize = @max(1, @as(usize, @intFromFloat(@round(@as(f64, @floatFromInt(rgb.h)) * ratio))));
    const scale_x = 1.0 / (@as(f64, @floatFromInt(dw)) / @as(f64, @floatFromInt(rgb.w)));
    const scale_y = 1.0 / (@as(f64, @floatFromInt(dh)) / @as(f64, @floatFromInt(rgb.h)));
    const kx = pyRound(scale_x);
    const ky = pyRound(scale_y);
    const eps = std.math.floatEps(f64);
    const px = try a.alloc(u8, dw * dh * 3);
    errdefer a.free(px);
    if (@abs(scale_x - @as(f64, @floatFromInt(kx))) < eps and @abs(scale_y - @as(f64, @floatFromInt(ky))) < eps) {
        const fx: usize = @intCast(kx);
        const fy: usize = @intCast(ky);
        std.debug.assert(rgb.w == fx * dw and rgb.h == fy * dh);
        const scale: f32 = 1.0 / @as(f32, @floatFromInt(fx * fy));
        for (0..dh) |dy| for (0..dw) |dx| for (0..3) |c| {
            var sum: u32 = 0;
            for (0..fy) |sy| for (0..fx) |sx| {
                sum += rgb.px[((dy * fy + sy) * rgb.w + dx * fx + sx) * 3 + c];
            };
            px[(dy * dw + dx) * 3 + c] = if (fx == 2 and fy == 2) @intCast((sum + 2) >> 2) else saturate8(@as(f32, @floatFromInt(sum)) * scale);
        };
        return .{ .w = dw, .h = dh, .px = px };
    }
    const xtab = try areaWeights(a, rgb.w, dw, 3, scale_x);
    defer a.free(xtab);
    const ytab = try areaWeights(a, rgb.h, dh, 1, scale_y);
    defer a.free(ytab);
    const buf = try a.alloc(f32, dw * 3);
    defer a.free(buf);
    const sum = try a.alloc(f32, dw * 3);
    defer a.free(sum);
    @memset(sum, 0);
    var prev = ytab[0].di;
    for (ytab) |y| {
        @memset(buf, 0);
        const row = rgb.px[y.si * rgb.w * 3 ..][0 .. rgb.w * 3];
        for (xtab) |x| for (0..3) |c| {
            buf[x.di + c] = buf[x.di + c] + @as(f32, @floatFromInt(row[x.si + c])) * x.alpha;
        };
        if (y.di != prev) {
            for (sum, px[prev * dw * 3 ..][0 .. dw * 3]) |v, *o| o.* = saturate8(v);
            for (sum, buf) |*v, b| v.* = y.alpha * b;
            prev = y.di;
        } else for (sum, buf) |*v, b| {
            v.* = v.* + y.alpha * b;
        }
    }
    for (sum, px[prev * dw * 3 ..][0 .. dw * 3]) |v, *o| o.* = saturate8(v);
    return .{ .w = dw, .h = dh, .px = px };
}

// ---------------------------------------------------------------------------------------------- PIL resample

const precision_bits = 32 - 8 - 2;

fn bicubic(x0: f64) f64 {
    const a = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/// precompute_coeffs (Resample.c) with BICUBIC's support 2: each output pixel's first input pixel, its count and
/// fixed-point weights (normalize_coeffs_8bpc).
const Coeffs = struct {
    ksize: usize,
    bounds: []usize, // [out][2]: xmin, count
    k: []i32, // [out][ksize]

    fn init(a: Allocator, in_size: usize, in0: f64, in1: f64, out_size: usize) !Coeffs {
        const support0 = 2.0;
        const scale = (in1 - in0) / @as(f64, @floatFromInt(out_size));
        const filterscale = @max(scale, 1.0);
        const support = support0 * filterscale;
        const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
        const bounds = try a.alloc(usize, out_size * 2);
        errdefer a.free(bounds);
        const k = try a.alloc(i32, out_size * ksize);
        errdefer a.free(k);
        const kk = try a.alloc(f64, ksize);
        defer a.free(kk);
        for (0..out_size) |xx| {
            const center = in0 + (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
            const ss = 1.0 / filterscale;
            // (int)(center - support + 0.5), clamped: C truncation toward zero
            var xmin_i: i64 = @intFromFloat(@trunc(center - support + 0.5));
            if (xmin_i < 0) xmin_i = 0;
            var xmax_i: i64 = @intFromFloat(@trunc(center + support + 0.5));
            if (xmax_i > @as(i64, @intCast(in_size))) xmax_i = @intCast(in_size);
            xmax_i -= xmin_i;
            const xmin: usize = @intCast(xmin_i);
            const xmax: usize = @intCast(@max(xmax_i, 0));
            var ww: f64 = 0.0;
            for (0..xmax) |x| {
                const wv = bicubic((@as(f64, @floatFromInt(x + xmin)) - center + 0.5) * ss);
                kk[x] = wv;
                ww += wv;
            }
            for (0..xmax) |x| {
                if (ww != 0.0) kk[x] /= ww;
            }
            for (xmax..ksize) |x| kk[x] = 0;
            for (0..ksize) |x| {
                const v = kk[x] * @as(f64, @floatFromInt(@as(i64, 1) << precision_bits));
                // (int)(-0.5 + v) for negatives, (int)(0.5 + v) otherwise: truncation toward zero
                k[xx * ksize + x] = @intFromFloat(@trunc(if (kk[x] < 0) -0.5 + v else 0.5 + v));
            }
            bounds[2 * xx] = xmin;
            bounds[2 * xx + 1] = xmax;
        }
        return .{ .ksize = ksize, .bounds = bounds, .k = k };
    }

    fn deinit(c: *Coeffs, a: Allocator) void {
        a.free(c.bounds);
        a.free(c.k);
    }
};

/// clip8: the sum's integer part (an arithmetic shift) clamped to 0..255.
fn clip8(ss: i64) u8 {
    const v = ss >> precision_bits;
    return @intCast(std.math.clamp(v, 0, 255));
}

/// One BICUBIC pass over RGB rows: horizontal (input rows y0.. of the source, out w_out x h_out) or vertical.
fn passH(src: Rgb, y0: usize, out: *Rgb, c: Coeffs) void {
    for (0..out.h) |yy| {
        const row = src.px[(yy + y0) * src.w * 3 ..];
        for (0..out.w) |xx| {
            const xmin = c.bounds[2 * xx];
            const xmax = c.bounds[2 * xx + 1];
            const k = c.k[xx * c.ksize ..];
            var s: [3]i64 = @splat(@as(i64, 1) << (precision_bits - 1));
            for (0..xmax) |x| {
                for (0..3) |ch| s[ch] += @as(i64, row[(x + xmin) * 3 + ch]) * k[x];
            }
            for (0..3) |ch| out.px[(yy * out.w + xx) * 3 + ch] = clip8(s[ch]);
        }
    }
}

fn passV(src: Rgb, out: *Rgb, c: Coeffs) void {
    for (0..out.h) |yy| {
        const ymin = c.bounds[2 * yy];
        const ymax = c.bounds[2 * yy + 1];
        const k = c.k[yy * c.ksize ..];
        for (0..out.w) |xx| {
            var s: [3]i64 = @splat(@as(i64, 1) << (precision_bits - 1));
            for (0..ymax) |y| {
                for (0..3) |ch| s[ch] += @as(i64, src.px[((y + ymin) * src.w + xx) * 3 + ch]) * k[y];
            }
            for (0..3) |ch| out.px[(yy * out.w + xx) * 3 + ch] = clip8(s[ch]);
        }
    }
}

/// ImagingResampleInner with BICUBIC over the box (x0, y0, x1, y1) of the source: the horizontal pass first (on the
/// rows the vertical one reads), then the vertical one, each only when that side changes (Resample.c's need_horizontal
/// and need_vertical, compared as it compares them).
fn resampleBox(a: Allocator, src: Rgb, xsize: usize, ysize: usize, box: [4]f64) !Rgb {
    const need_h = xsize != src.w or box[0] != 0 or box[2] != @as(f64, @floatFromInt(xsize));
    const need_v = ysize != src.h or box[1] != 0 or box[3] != @as(f64, @floatFromInt(ysize));
    var ch = try Coeffs.init(a, src.w, box[0], box[2], xsize);
    defer ch.deinit(a);
    var cv = try Coeffs.init(a, src.h, box[1], box[3], ysize);
    defer cv.deinit(a);
    const ybox_first = cv.bounds[0];
    const ybox_last = cv.bounds[ysize * 2 - 2] + cv.bounds[ysize * 2 - 1];
    var cur = src;
    var owned: ?Rgb = null;
    errdefer if (owned) |*o| o.deinit(a);
    if (need_h) {
        for (0..ysize) |i| cv.bounds[i * 2] -= ybox_first;
        var tmp: Rgb = .{ .w = xsize, .h = ybox_last - ybox_first, .px = try a.alloc(u8, xsize * (ybox_last - ybox_first) * 3) };
        passH(src, ybox_first, &tmp, ch);
        owned = tmp;
        cur = tmp;
    }
    if (need_v) {
        var out: Rgb = .{ .w = cur.w, .h = ysize, .px = try a.alloc(u8, cur.w * ysize * 3) };
        passV(cur, &out, cv);
        if (owned) |*o| o.deinit(a);
        return out;
    }
    if (owned) |o| return o;
    return .{ .w = src.w, .h = src.h, .px = try a.dupe(u8, src.px) };
}

/// Image.resize(size, BICUBIC) of an RGB image (the whole source as the box), with PIL's two steps for a source more
/// than 100 times taller than wide being shortened.
pub fn resize(a: Allocator, src: Rgb, w: usize, h: usize) !Rgb {
    if (w == src.w and h == src.h) return .{ .w = w, .h = h, .px = try a.dupe(u8, src.px) };
    const fw: f64 = @floatFromInt(src.w);
    const fh: f64 = @floatFromInt(src.h);
    if (src.h > src.w * 100 and h < src.h) {
        var tall = try resampleBox(a, src, src.w, h, .{ 0, 0, fw, fh });
        defer tall.deinit(a);
        return resampleBox(a, tall, w, h, .{ 0, 0, fw, @floatFromInt(h) });
    }
    return resampleBox(a, src, w, h, .{ 0, 0, fw, fh });
}

/// ImageOps.pad(image, (w, h), color=(127, 127, 127)): contain() (the aspect kept, Python's round of the scaled side),
/// then pasted centred (Python's round of half the gap) on a grey canvas.
pub fn pad(a: Allocator, src: Rgb, w: usize, h: usize) !Rgb {
    const im_ratio = @as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h));
    const dest_ratio = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var cw = w;
    var chh = h;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            const nh: usize = @intCast(pyRound(@as(f64, @floatFromInt(src.h)) / @as(f64, @floatFromInt(src.w)) * @as(f64, @floatFromInt(w))));
            if (nh != h) chh = nh;
        } else {
            const nw: usize = @intCast(pyRound(@as(f64, @floatFromInt(src.w)) / @as(f64, @floatFromInt(src.h)) * @as(f64, @floatFromInt(h))));
            if (nw != w) cw = nw;
        }
    }
    var fit = try resize(a, src, cw, chh);
    if (fit.w == w and fit.h == h) return fit;
    defer fit.deinit(a);
    const px = try a.alloc(u8, w * h * 3);
    @memset(px, 127);
    var x0: usize = 0;
    var y0: usize = 0;
    if (fit.w != w) {
        x0 = @intCast(pyRound(@as(f64, @floatFromInt(w - fit.w)) * 0.5));
    } else {
        y0 = @intCast(pyRound(@as(f64, @floatFromInt(h - fit.h)) * 0.5));
    }
    for (0..fit.h) |y| {
        @memcpy(px[((y + y0) * w + x0) * 3 ..][0 .. fit.w * 3], fit.px[y * fit.w * 3 ..][0 .. fit.w * 3]);
    }
    return .{ .w = w, .h = h, .px = px };
}

/// f32 to bf16, round to nearest even (torch's .to(torch.bfloat16)).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}

/// The ViT's input of a padded image: n_vit_h * n_vit_w patches of [3][patch][patch] bf16, each value
/// (v / 255 - 0.5) / 0.5 in f32 (the reference's numpy / torch float32 steps).
pub fn patchify(a: Allocator, img: Rgb, p: usize) ![]u16 {
    const nh = img.h / p;
    const nw = img.w / p;
    const out = try a.alloc(u16, nh * nw * 3 * p * p);
    var lut: [256]u16 = undefined;
    for (0..256) |v| {
        const x = @as(f32, @floatFromInt(v)) / 255.0;
        lut[v] = bf16((x - 0.5) / 0.5);
    }
    for (0..nh) |ph| for (0..nw) |pw| {
        const base = (ph * nw + pw) * 3 * p * p;
        for (0..3) |c| for (0..p) |py| for (0..p) |px| {
            const y = ph * p + py;
            const x = pw * p + px;
            out[base + (c * p + py) * p + px] = lut[img.px[(y * img.w + x) * 3 + c]];
        };
    };
    return out;
}

/// A picture: its grid and its patches (owned).
pub const Picture = struct {
    grid: Grid,
    patches: []u16,

    pub fn deinit(p: *Picture, a: Allocator) void {
        a.free(p.patches);
        p.* = undefined;
    }
};

/// decode (load_image): bytes to the ViT's patches and the grids.
pub fn picture(a: Allocator, data: []const u8, cfg: Config) !Picture {
    var rgb = try decodePng(a, data);
    defer rgb.deinit(a);
    return fromRgb(a, rgb, cfg);
}

/// load_image after its decode: the decoded RGB planned, padded and cut into the ViT's patches.
pub fn fromRgb(a: Allocator, rgb: Rgb, cfg: Config) !Picture {
    const g = plan(rgb.w, rgb.h, cfg) orelse return error.NoGrid;
    var padded = try pad(a, rgb, g.best_w, g.best_h);
    defer padded.deinit(a);
    return .{ .grid = g, .patches = try patchify(a, padded, cfg.patch) };
}

/// The planned RGB image as its span length and grid header followed by bf16 patches.
pub fn prepare(a: Allocator, rgb: Rgb, cfg: Config) !Prepared {
    var pic = try fromRgb(a, rgb, cfg);
    defer pic.deinit(a);
    const g = pic.grid;
    const payload = try a.alignedAlloc(u8, .@"16", header_bytes + pic.patches.len * 2);
    @memset(payload[0..header_bytes], 0);
    for ([_]usize{ g.best_h, g.best_w, g.n_vit_h, g.n_vit_w, g.n_llm_h, g.n_llm_w }, 0..) |x, i| std.mem.writeInt(u32, payload[i * 4 ..][0..4], @intCast(x), .little);
    @memcpy(payload[header_bytes..], std.mem.sliceAsBytes(pic.patches));
    return .{ .tokens = @intCast(g.spanTokens()), .payload = payload };
}

test "grids of the vision gate's images as the reference plans them" {
    const cfg: Config = .{};
    // (w, h) -> best_w, best_h, n_vit_h, n_vit_w, n_llm_h, n_llm_w (py/vfix.py on the reference)
    const cases = [_][8]usize{
        .{ 560, 400, 644, 462, 33, 46, 11, 16 },
        .{ 420, 520, 490, 616, 44, 35, 15, 12 },
        .{ 400, 300, 630, 476, 34, 45, 12, 15 },
        .{ 120, 2000, 140, 2226, 159, 10, 53, 4 },
        .{ 640, 240, 896, 336, 24, 64, 8, 22 },
        .{ 16, 16, 546, 546, 39, 39, 13, 13 },
        .{ 3000, 90, 3150, 98, 7, 225, 3, 75 },
    };
    for (cases) |c| {
        const g = plan(c[0], c[1], cfg).?;
        try std.testing.expectEqual(c[2], g.best_w);
        try std.testing.expectEqual(c[3], g.best_h);
        try std.testing.expectEqual(c[4], g.n_vit_h);
        try std.testing.expectEqual(c[5], g.n_vit_w);
        try std.testing.expectEqual(c[6], g.n_llm_h);
        try std.testing.expectEqual(c[7], g.n_llm_w);
    }
}

test "grids repeat until they hold, as DeepSeek Recipe's calc_resize" {
    const cfg: Config = .{};
    // (w, h) -> best_w, best_h, n_llm_h, n_llm_w, tokens (token_spec.rs CASES; 1x161 and 1x352 a single pass plans short)
    const cases = [_][7]usize{
        .{ 2, 165, 70, 4942, 118, 2, 356 },
        .{ 165, 2, 4942, 70, 2, 118, 240 },
        .{ 158, 10000, 126, 7966, 190, 3, 762 },
        .{ 10000, 158, 10010, 168, 4, 239, 962 },
        .{ 1, 1, 546, 546, 13, 13, 184 },
        .{ 16809, 11841, 1540, 1092, 26, 37, 990 },
        .{ 1920, 1080, 1708, 966, 23, 41, 968 },
        .{ 125, 751, 224, 1344, 32, 6, 226 },
        .{ 1, 161, 42, 7056, 168, 1, 338 },
        .{ 1, 352, 28, 10570, 252, 1, 506 },
    };
    for (cases) |c| {
        const g = plan(c[0], c[1], cfg).?;
        try std.testing.expectEqual(c[2], g.best_w);
        try std.testing.expectEqual(c[3], g.best_h);
        try std.testing.expectEqual(c[4], g.n_llm_h);
        try std.testing.expectEqual(c[5], g.n_llm_w);
        try std.testing.expectEqual(c[6], g.spanTokens());
    }
}

test "transparency blends onto #FDFDFD and 16-bit samples divide by 257, as OpenCV decodes them" {
    try std.testing.expectEqual(@as(u8, 255), from16(65535));
    try std.testing.expectEqual(@as(u8, 0), from16(128));
    try std.testing.expectEqual(@as(u8, 1), from16(129));
    try std.testing.expectEqual(@as(u8, 1), from16(385));
    try std.testing.expectEqual(@as(u8, 2), from16(386));
    const a = std.testing.allocator;
    // 2x1 RGBA at 16 bits: (65535, 0, 32768) at alpha 32768, then all zero
    const rgba16 = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x02\x00\x00\x00\x01\x10\x06\x00\x00\x00\xa4\xb2\xa3\xc9\x00\x00\x00\x11\x49\x44\x41\x54\x78\xda\x63\xf8\xff\x9f\x81\xa1\x01\x08\x61\x00\x00\x29\xf2\x02\xff\xd4\x40\x35\x50\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    var x = try decodePng(a, rgba16);
    defer x.deinit(a);
    try std.testing.expectEqualSlices(u8, &.{ 254, 126, 190, 253, 253, 253 }, x.px);
    // 2x1 palette (10, 20, 30), (200, 100, 50), tRNS 0 for the first entry
    const pal = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x02\x00\x00\x00\x01\x08\x03\x00\x00\x00\xc3\xfc\x8f\xb8\x00\x00\x00\x06\x50\x4c\x54\x45\x0a\x14\x1e\xc8\x64\x32\x77\xa0\xb3\x9c\x00\x00\x00\x01\x74\x52\x4e\x53\x00\x40\xe6\xd8\x66\x00\x00\x00\x0b\x49\x44\x41\x54\x78\xda\x63\x60\x60\x04\x00\x00\x04\x00\x02\x2c\xde\x48\xad\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    var y = try decodePng(a, pal);
    defer y.deinit(a);
    try std.testing.expectEqualSlices(u8, &.{ 253, 253, 253, 200, 100, 50 }, y.px);
}

test "low detail scales the long side down as OpenCV's INTER_AREA does on x86" {
    const a = std.testing.allocator;
    // pixels (i * 37 + 11) % 256; expected values from opencv-python-headless 4.12.0 on x86_64
    var src: [9 * 4 * 3]u8 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast((i * 37 + 11) % 256);
    try std.testing.expectEqual(@as(?Rgb, null), try limitLongSide(a, .{ .w = 9, .h = 4, .px = &src }, 9));
    var block = (try limitLongSide(a, .{ .w = 4, .h = 2, .px = src[0 .. 4 * 2 * 3] }, 2)).?;
    defer block.deinit(a);
    try std.testing.expectEqual(@as(usize, 2), block.w);
    try std.testing.expectEqual(@as(usize, 1), block.h);
    try std.testing.expectEqualSlices(u8, &.{ 97, 134, 107, 127, 100, 137 }, block.px);
    var weighed = (try limitLongSide(a, .{ .w = 9, .h = 4, .px = &src }, 4)).?;
    defer weighed.deinit(a);
    try std.testing.expectEqual(@as(usize, 4), weighed.w);
    try std.testing.expectEqual(@as(usize, 2), weighed.h);
    try std.testing.expectEqualSlices(u8, &.{ 129, 95, 118, 148, 143, 109, 111, 148, 156, 130, 110, 147, 136, 173, 82, 98, 136, 130, 118, 98, 135, 137, 117, 97 }, weighed.px);
}

test "Python's round: half to even" {
    try std.testing.expectEqual(@as(i64, 2), pyRound(2.5));
    try std.testing.expectEqual(@as(i64, 4), pyRound(3.5));
    try std.testing.expectEqual(@as(i64, 3), pyRound(2.6));
    try std.testing.expectEqual(@as(i64, 0), pyRound(0.5));
}

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0xbf80), bf16(-1.0));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(@bitCast(@as(u32, 0x3f80c000)))); // above half: up
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // half, even: down
}

test "a span's token types" {
    const g: Grid = .{ .best_h = 0, .best_w = 0, .n_vit_h = 0, .n_vit_w = 0, .n_llm_h = 2, .n_llm_w = 3 };
    var t: [10]Type = undefined;
    spanTypes(g, &t);
    try std.testing.expectEqualSlices(Type, &.{ .start, .image, .image, .image, .newline, .image, .image, .image, .newline, .end }, &t);
}
