//! Pictures picture.zig does not decode itself (every format but the PNGs it reads: JPEG, WebP, GIF, BMP, TIFF, AVIF,
//! ICO, PPM, ...), decoded by Pillow's Image.open(BytesIO(data)) as the served lane opens them, in a long-lived
//! `python3 -I` child with the lane's pixel limit, one image at a time, and made 8-bit RGB as DeepSeek Recipe's
//! preprocessing makes it: an image with alpha or a transparency entry as Pillow's RGBA blended onto #FDFDFD, 16-bit
//! grey divided by 257, any other as Pillow's convert("RGB") (the DeepSeek reference's load_image). Without python3
//! and Pillow the child is absent and such images are refused as Pillow refuses data it cannot identify.
const std = @import("std");
const picture = @import("picture.zig");
const Allocator = std.mem.Allocator;

/// The child: a request is the image's length (u64 LE) and bytes; the answer a status byte (0: RGB rows, 1: Pillow's
/// DecompressionBombError, 2: another error, 3: RGBA rows, 4: 16-bit grey rows, little-endian), width, height and the
/// length of what follows (u64 LE each), then the rows or the error's text. The lane refuses past 134,217,728 pixels:
/// twice Image.MAX_IMAGE_PIXELS.
pub const script =
    \\import array, io, struct, sys, warnings
    \\from PIL import Image
    \\warnings.simplefilter("ignore")
    \\Image.MAX_IMAGE_PIXELS = 67108864
    \\inp, out = sys.stdin.buffer, sys.stdout.buffer
    \\def take(n):
    \\    b = inp.read(n)
    \\    if len(b) < n:
    \\        sys.exit(0)
    \\    return b
    \\out.write(b"PIL" + Image.__version__.encode().ljust(13))
    \\out.flush()
    \\while True:
    \\    data = take(struct.unpack("<Q", take(8))[0])
    \\    try:
    \\        with Image.open(io.BytesIO(data)) as source:
    \\            size = source.size
    \\            if source.mode in ("I;16", "I;16L", "I;16B", "I;16N"):
    \\                kind, px = 4, source.tobytes()
    \\                if source.mode == "I;16B" or (source.mode == "I;16N" and sys.byteorder == "big"):
    \\                    swapped = array.array("H", px)
    \\                    swapped.byteswap()
    \\                    px = swapped.tobytes()
    \\            elif "transparency" in source.info or source.mode in ("RGBA", "RGBa", "LA", "La", "PA"):
    \\                kind, px = 3, source.convert("RGBA").tobytes()
    \\            else:
    \\                kind, px = 0, source.convert("RGB").tobytes()
    \\        out.write(struct.pack("<BQQQ", kind, size[0], size[1], len(px)) + px)
    \\    except Image.DecompressionBombError as e:
    \\        m = str(e).encode()
    \\        out.write(struct.pack("<BQQQ", 1, 0, 0, len(m)) + m)
    \\    except Exception as e:
    \\        m = str(e).encode()
    \\        out.write(struct.pack("<BQQQ", 2, 0, 0, len(m)) + m)
    \\    out.flush()
;

pub const Pil = struct {
    gpa: Allocator,
    io: std.Io,
    child: std.process.Child,
    mutex: std.Io.Mutex = .init,
    reader: std.Io.File.Reader,
    buf: [64 << 10]u8 = undefined,
    /// Pillow's version, as the child reports it on start.
    version: [13]u8 = undefined,

    /// The child started and its Pillow named; error when there is no python3 or no Pillow.
    pub fn start(gpa: Allocator, io: std.Io) !*Pil {
        const p = try gpa.create(Pil);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .io = io, .child = undefined, .reader = undefined };
        try p.spawn();
        return p;
    }

    /// The child, its reader, and Pillow's greeting.
    fn spawn(p: *Pil) !void {
        p.child = try std.process.spawn(p.io, .{ .argv = &.{ "python3", "-I", "-c", script }, .stdin = .pipe, .stdout = .pipe, .stderr = .inherit });
        errdefer p.child.kill(p.io);
        p.reader = p.child.stdout.?.readerStreaming(p.io, &p.buf);
        var hello: [16]u8 = undefined;
        p.reader.interface.readSliceAll(&hello) catch return error.NoPillow;
        if (!std.mem.eql(u8, hello[0..3], "PIL")) return error.NoPillow;
        @memcpy(&p.version, hello[3..16]);
    }

    pub fn stop(p: *Pil, gpa: Allocator) void {
        if (p.child.stdin) |f| f.close(p.io);
        p.child.stdin = null;
        _ = p.child.wait(p.io) catch {};
        gpa.destroy(p);
    }

    pub fn versionText(p: *const Pil) []const u8 {
        return std.mem.trimEnd(u8, &p.version, " ");
    }

    pub const Decoded = union(enum) { rgb: picture.Rgb, refused: []const u8 };

    /// The image's 8-bit RGB, or the lane's refusal of it (`a` holds either); a child that has died is started again
    /// once.
    pub fn decode(p: *Pil, a: Allocator, data: []const u8) !Decoded {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        return p.exchange(a, data) catch |err| switch (err) {
            error.OutOfMemory, error.BadPillowReply => err,
            else => {
                std.log.warn("Pillow's child failed ({s}): starting it again", .{@errorName(err)});
                p.child.kill(p.io);
                try p.spawn();
                return p.exchange(a, data);
            },
        };
    }

    fn exchange(p: *Pil, a: Allocator, data: []const u8) !Decoded {
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, data.len, .little);
        const in = p.child.stdin orelse return error.NoPillow;
        try in.writeStreamingAll(p.io, &len);
        try in.writeStreamingAll(p.io, data);
        var head: [25]u8 = undefined;
        try p.reader.interface.readSliceAll(&head);
        const w: usize = @intCast(std.mem.readInt(u64, head[1..9], .little));
        const h: usize = @intCast(std.mem.readInt(u64, head[9..17], .little));
        const n: usize = @intCast(std.mem.readInt(u64, head[17..25], .little));
        const per: usize = switch (head[0]) {
            0 => 3,
            3 => 4,
            4 => 2,
            else => 0,
        };
        if (per != 0 and n != w * h * per) return error.BadPillowReply;
        const body = try a.alloc(u8, n);
        errdefer a.free(body);
        try p.reader.interface.readSliceAll(body);
        switch (head[0]) {
            0 => return .{ .rgb = .{ .w = w, .h = h, .px = body } },
            1 => return .{ .refused = body },
            3, 4 => {},
            else => return .{ .refused = try std.fmt.allocPrint(a, "image input: {s}", .{body}) },
        }
        const px = try a.alloc(u8, w * h * 3);
        for (0..w * h) |i| {
            const rgb: [3]u8 = if (head[0] == 3) blk: {
                const s = body[i * 4 ..][0..4];
                break :blk .{ picture.blend(s[0], s[3]), picture.blend(s[1], s[3]), picture.blend(s[2], s[3]) };
            } else blk: {
                const v = picture.from16(std.mem.readInt(u16, body[i * 2 ..][0..2], .little));
                break :blk .{ v, v, v };
            };
            px[i * 3 ..][0..3].* = rgb;
        }
        a.free(body);
        return .{ .rgb = .{ .w = w, .h = h, .px = px } };
    }
};
