//! Kept prompt states' compressed rows on disk. With TF_DS_SPILL_DIR set (at most TF_DS_SPILL_GIB, default 48), a
//! kept state whose extent a new stream needs is written out instead of going stale, and when its prompt comes back
//! its rows are read into a free extent: a long agent session resumes after other sessions crowded it out of the pool.
//! Each rank writes its own rows (rank 1 on rank 0's snap_spill frame); a state's file is removed when it is dropped.
const std = @import("std");
const Model = @import("model.zig").Model;

pub const Spill = struct {
    dir: []const u8 = "",
    cap: u64 = 0,
    used: u64 = 0,
    rank: u32 = 0,
    bounce: []align(4096) u8 = &.{},
    files: std.AutoHashMapUnmanaged(u64, u64) = .empty, // state id -> its file's bytes

    pub const piece_bytes = 64 << 20;

    pub fn fromEnv(gpa: std.mem.Allocator, rank: u32) Spill {
        const d = std.c.getenv("TF_DS_SPILL_DIR") orelse return .{};
        const dir = std.mem.span(d);
        if (dir.len == 0) return .{};
        const gib: u64 = if (std.c.getenv("TF_DS_SPILL_GIB")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 48 else 48;
        const b = gpa.alignedAlloc(u8, .fromByteUnits(4096), piece_bytes) catch return .{};
        std.log.info("spill: kept prompts' rows to {s} (at most {d} GiB)", .{ dir, gib });
        return .{ .dir = dir, .cap = gib << 30, .rank = rank, .bounce = b };
    }

    pub fn on(s: *const Spill) bool {
        return s.dir.len > 0 and s.bounce.len > 0;
    }

    pub fn deinit(s: *Spill, gpa: std.mem.Allocator) void {
        var it = s.files.keyIterator();
        while (it.next()) |id| s.unlinkFile(id.*);
        s.files.deinit(gpa);
        if (s.bounce.len > 0) gpa.free(s.bounce);
        s.* = .{};
    }

    fn path(s: *const Spill, buf: []u8, id: u64) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/spill-r{d}-{d}.kv", .{ s.dir, s.rank, id }, 0);
    }

    fn unlinkFile(s: *const Spill, id: u64) void {
        var pb: [1024]u8 = undefined;
        const p = s.path(&pb, id) catch return;
        _ = std.c.unlink(p);
    }

    pub fn has(s: *const Spill, id: u64) bool {
        return s.files.contains(id);
    }

    /// State `id`'s rows (the extent at `base`, positions [0, at)) written to its file; error.SpillFull past the cap.
    pub fn write(s: *Spill, gpa: std.mem.Allocator, m: *Model, id: u64, base: usize, at: usize) !void {
        if (!s.on()) return error.SpillOff;
        if (s.files.contains(id)) return; // its rows never change while it lives: the file stands
        const est = m.extentBytes(at) + 64 * 4096;
        if (s.used + est > s.cap) return error.SpillFull;
        var pb: [1024]u8 = undefined;
        const p = try s.path(&pb, id);
        var fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true, .DIRECT = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.SpillOpen;
        defer _ = std.c.close(fd);
        const bytes = m.diskExtent(base, at, fd, s.bounce, true) catch |err| {
            _ = std.c.unlink(p);
            return err;
        };
        try s.files.put(gpa, id, bytes);
        s.used += bytes;
    }

    /// State `id`'s rows read from its file into the extent at `to`.
    pub fn read(s: *Spill, m: *Model, id: u64, to: usize, at: usize) !void {
        if (!s.files.contains(id)) return error.NotSpilled;
        var pb: [1024]u8 = undefined;
        const p = try s.path(&pb, id);
        var fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true }, @as(std.c.mode_t, 0));
        if (fd < 0) fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SpillOpen;
        defer _ = std.c.close(fd);
        _ = try m.diskExtent(to, at, fd, s.bounce, false);
    }

    /// State `id`'s file removed (the state is dropped).
    pub fn forget(s: *Spill, id: u64) void {
        const kv = s.files.fetchRemove(id) orelse return;
        s.used -|= kv.value;
        s.unlinkFile(id);
    }
};
