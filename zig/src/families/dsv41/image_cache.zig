const std = @import("std");
const picture = @import("picture.zig");
const Allocator = std.mem.Allocator;

pub const Key = [std.crypto.hash.sha2.Sha256.digest_length]u8;

pub fn key(bytes: []const u8, low_detail: bool, cfg: picture.Config) Key {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    h.update(&.{@intFromBool(low_detail)});
    inline for (@typeInfo(picture.Config).@"struct".field_names) |name| {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, @intCast(@field(cfg, name)), .little);
        h.update(&buf);
    }
    return h.finalResult();
}

pub fn limitBytes(text: ?[]const u8) error{BadImageCache}!usize {
    const mib = std.fmt.parseInt(usize, text orelse return 0, 10) catch return error.BadImageCache;
    return std.math.mul(usize, mib, 1 << 20) catch error.BadImageCache;
}

/// Prepared images in least-recently-used order; hits return independent, 16-byte-aligned copies.
pub const Cache = struct {
    gpa: Allocator,
    io: std.Io,
    max_bytes: usize,
    used: usize = 0,
    mutex: std.Io.Mutex = .init,
    entries: std.DoublyLinkedList = .{},

    const Entry = struct { node: std.DoublyLinkedList.Node = .{}, key: Key, data: picture.Prepared };

    pub fn init(gpa: Allocator, io: std.Io, max_bytes: usize) Cache {
        return .{ .gpa = gpa, .io = io, .max_bytes = max_bytes };
    }

    pub fn deinit(c: *Cache) void {
        while (c.entries.first) |n| c.drop(@fieldParentPtr("node", n));
        c.* = undefined;
    }

    fn find(c: *Cache, k: Key) ?*Entry {
        var node = c.entries.first;
        while (node) |n| : (node = n.next) {
            const e: *Entry = @fieldParentPtr("node", n);
            if (std.mem.eql(u8, &e.key, &k)) return e;
        }
        return null;
    }

    fn drop(c: *Cache, e: *Entry) void {
        c.entries.remove(&e.node);
        c.used -= @sizeOf(Entry) + e.data.payload.len;
        c.gpa.free(e.data.payload);
        c.gpa.destroy(e);
    }

    pub fn get(c: *Cache, a: Allocator, k: Key) Allocator.Error!?picture.Prepared {
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        const e = c.find(k) orelse return null;
        const payload = try a.alignedAlloc(u8, .@"16", e.data.payload.len);
        @memcpy(payload, e.data.payload);
        c.entries.remove(&e.node);
        c.entries.append(&e.node);
        return .{ .tokens = e.data.tokens, .payload = payload };
    }

    pub fn put(c: *Cache, k: Key, data: picture.Prepared) Allocator.Error!void {
        if (c.max_bytes < @sizeOf(Entry) or data.payload.len > c.max_bytes - @sizeOf(Entry)) return;
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        if (c.find(k)) |e| {
            c.entries.remove(&e.node);
            c.entries.append(&e.node);
            return;
        }
        const bytes = @sizeOf(Entry) + data.payload.len;
        while (c.used > c.max_bytes - bytes) c.drop(@fieldParentPtr("node", c.entries.first.?));
        const e = try c.gpa.create(Entry);
        errdefer c.gpa.destroy(e);
        const payload = try c.gpa.alignedAlloc(u8, .@"16", data.payload.len);
        @memcpy(payload, data.payload);
        e.* = .{ .key = k, .data = .{ .tokens = data.tokens, .payload = payload } };
        c.entries.append(&e.node);
        c.used += bytes;
    }
};

test "image keys include the original bytes, the detail and every picture setting" {
    const cfg: picture.Config = .{};
    const k = key("image", false, cfg);
    try std.testing.expectEqual(k, key("image", false, cfg));
    try std.testing.expect(!std.mem.eql(u8, &k, &key("other", false, cfg)));
    try std.testing.expect(!std.mem.eql(u8, &k, &key("image", true, cfg)));
    inline for (@typeInfo(picture.Config).@"struct".field_names) |name| {
        var other = cfg;
        @field(other, name) += 1;
        try std.testing.expect(!std.mem.eql(u8, &k, &key("image", false, other)));
    }
}

test "image cache hits keep every prepared byte and its alignment" {
    const a = std.testing.allocator;
    const px = try a.dupe(u8, &.{ 0, 127, 255, 3, 11, 21, 99, 180, 240, 255, 0, 128 });
    var rgb: picture.Rgb = .{ .w = 2, .h = 2, .px = px };
    defer rgb.deinit(a);
    const cfg: picture.Config = .{ .min_pixels = 0 };
    const fresh = try picture.prepare(a, rgb, cfg);
    defer a.free(fresh.payload);
    var c = Cache.init(a, std.testing.io, 1 << 20);
    defer c.deinit();
    const k = key(px, false, cfg);
    try std.testing.expectEqual(@as(?picture.Prepared, null), try c.get(a, k));
    try c.put(k, fresh);
    const hit = (try c.get(a, k)).?;
    defer a.free(hit.payload);
    try std.testing.expectEqual(fresh.tokens, hit.tokens);
    try std.testing.expectEqualSlices(u8, fresh.payload, hit.payload);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(hit.payload.ptr) % 16);
    try std.testing.expect(hit.payload.ptr != fresh.payload.ptr);
    const used = c.used;
    try c.put(k, fresh);
    try std.testing.expectEqual(used, c.used);
}

test "image cache evicts the least recent entry and keeps returned copies alive" {
    const a = std.testing.allocator;
    const bytes align(16) = [_]u8{ 1, 2, 3, 4 };
    const data: picture.Prepared = .{ .tokens = 7, .payload = &bytes };
    const size = @sizeOf(Cache.Entry) + bytes.len;
    var c = Cache.init(a, std.testing.io, 2 * size);
    defer c.deinit();
    const k1 = key("one", false, .{});
    const k2 = key("two", false, .{});
    const k3 = key("three", false, .{});
    try c.put(k1, data);
    try c.put(k2, data);
    const kept = (try c.get(a, k1)).?;
    defer a.free(kept.payload);
    try c.put(k3, data);
    try std.testing.expectEqual(@as(?picture.Prepared, null), try c.get(a, k2));
    try c.put(k2, data);
    try std.testing.expectEqual(@as(?picture.Prepared, null), try c.get(a, k1));
    try std.testing.expectEqualSlices(u8, &bytes, kept.payload);
    try std.testing.expectEqual(2 * size, c.used);
}

test "image cache budgets disable storage and refuse oversized entries" {
    const a = std.testing.allocator;
    const bytes align(16) = [_]u8{ 1, 2, 3, 4 };
    const data: picture.Prepared = .{ .tokens = 7, .payload = &bytes };
    const k = key("one", false, .{});
    for ([_]usize{ 0, @sizeOf(Cache.Entry) + bytes.len - 1 }) |limit| {
        var c = Cache.init(a, std.testing.io, limit);
        defer c.deinit();
        try c.put(k, data);
        try std.testing.expectEqual(@as(?picture.Prepared, null), try c.get(a, k));
        try std.testing.expectEqual(@as(usize, 0), c.used);
    }
    try std.testing.expectEqual(@as(usize, 0), try limitBytes(null));
    try std.testing.expectEqual(@as(usize, 0), try limitBytes("0"));
    try std.testing.expectEqual(@as(usize, 2 << 20), try limitBytes("2"));
    for ([_][]const u8{ "", "-1", "1.5", "18446744073709551615" }) |v| try std.testing.expectError(error.BadImageCache, limitBytes(v));
}

test "image cache requests can insert and copy hits concurrently" {
    const Worker = struct {
        cache: *Cache,
        id: u8,
        err: ?anyerror = null,

        fn run(w: *@This()) void {
            w.work() catch |err| {
                w.err = err;
            };
        }

        fn work(w: *@This()) !void {
            const bytes align(16) = [_]u8{ w.id, 2, 3, 4 };
            const k = key(&bytes, false, .{});
            for (0..32) |_| {
                try w.cache.put(k, .{ .tokens = w.id, .payload = &bytes });
                if (try w.cache.get(std.testing.allocator, k)) |hit| {
                    defer std.testing.allocator.free(hit.payload);
                    try std.testing.expectEqual(@as(u32, w.id), hit.tokens);
                    try std.testing.expectEqualSlices(u8, &bytes, hit.payload);
                }
            }
        }
    };
    var c = Cache.init(std.testing.allocator, std.testing.io, 2 * (@sizeOf(Cache.Entry) + 4));
    defer c.deinit();
    var workers: [4]Worker = undefined;
    var threads: [4]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |t| t.join();
    for (&workers, &threads, 0..) |*w, *t, i| {
        w.* = .{ .cache = &c, .id = @intCast(i) };
        t.* = try std.Thread.spawn(.{}, Worker.run, .{w});
        started += 1;
    }
    for (threads) |t| t.join();
    started = 0;
    for (workers) |w| if (w.err) |err| return err;
    try std.testing.expect(c.used <= c.max_bytes);
}
