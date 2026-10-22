const std = @import("std");

pub const Settings = struct {
    enabled: bool = false,
    min_rows: usize = 1024,

    pub fn plan(s: Settings, rows: usize, rb: usize, world: usize, encoder_source: bool) ?Mirror {
        if (!s.enabled or rows <= s.min_rows or world != 2 or !encoder_source) return null;
        return Mirror.init(rows, rb);
    }
};

/// Two ranks' mirrored quarters, with every boundary on an original row block.
pub const Mirror = struct {
    bounds: [5]usize,
    send_rows: usize,

    pub const Span = struct { row: usize, rows: usize, local: usize, gathered: usize };

    pub fn init(rows: usize, rb: usize) ?Mirror {
        if (rb == 0) return null;
        const blocks = rows / rb + @intFromBool(rows % rb != 0);
        if (blocks < 4) return null;
        var bounds: [5]usize = undefined;
        for (&bounds, 0..) |*b, j| b.* = @min(rows, ((blocks * j + 2) / 4) * rb);
        return .{ .bounds = bounds, .send_rows = @max(bounds[1] + rows - bounds[3], bounds[3] - bounds[1]) };
    }

    /// The rank's source rows and their offsets in its send buffer and the all-gather result.
    pub fn spans(m: Mirror, rank: usize) [2]Span {
        std.debug.assert(rank < 2);
        var out: [2]Span = undefined;
        var local: usize = 0;
        for (&out, 0..) |*sp, half| {
            const quarter = if (half == 0) rank else 3 - rank;
            sp.* = .{ .row = m.bounds[quarter], .rows = m.bounds[quarter + 1] - m.bounds[quarter], .local = local, .gathered = rank * m.send_rows + local };
            local += sp.rows;
        }
        return out;
    }
};

test "index split stays off outside enabled TP2 encoder chunks above the threshold" {
    try std.testing.expect((Settings{}).plan(4096, 32, 2, true) == null);
    const s: Settings = .{ .enabled = true };
    try std.testing.expect(s.plan(1024, 32, 2, true) == null);
    try std.testing.expect(s.plan(4096, 32, 4, true) == null);
    try std.testing.expect(s.plan(4096, 32, 2, false) == null);
    try std.testing.expect(s.plan(4096, 1366, 2, true) == null);
    try std.testing.expect(s.plan(4096, 0, 2, true) == null);
    try std.testing.expect(s.plan(1025, 32, 2, true) != null);
    try std.testing.expect((Settings{ .enabled = true, .min_rows = 4096 }).plan(4096, 32, 2, true) == null);
}

test "index split mirrors quarters and balances causal rows" {
    const m = Mirror.init(4096, 32).?;
    try std.testing.expectEqual([5]usize{ 0, 1024, 2048, 3072, 4096 }, m.bounds);
    try std.testing.expectEqual(@as(usize, 2048), m.send_rows);
    var work: [2]usize = @splat(0);
    for (0..2) |rank| for (m.spans(rank)) |sp| {
        for (sp.row..sp.row + sp.rows) |row| work[rank] += (row + 1) / 2;
    };
    try std.testing.expectEqual(work[0], work[1]);
}

test "index split preserves row blocks and merges padded selections byte for byte" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 3, 16, 17, 31, 32, 127, 512, 1023 }) |rb| {
        for ([_]usize{ 3 * rb + 1, 4 * rb, 4 * rb + 1, 6 * rb - 1, 7 * rb, 4095, 4096 }) |rows| {
            const m = Mirror.init(rows, rb) orelse continue;
            const kk = 7;
            const source = try a.alloc(i64, rows * kk);
            defer a.free(source);
            const gathered = try a.alloc(i64, 2 * m.send_rows * kk);
            defer a.free(gathered);
            const merged = try a.alloc(i64, rows * kk);
            defer a.free(merged);
            const owners = try a.alloc(u8, rows);
            defer a.free(owners);
            for (source, 0..) |*v, j| v.* = if (j % kk == kk - 1) -1 else @intCast(j);
            @memset(gathered, -1);
            @memset(merged, -2);
            @memset(owners, 0);
            for (0..2) |rank| for (m.spans(rank)) |sp| {
                try std.testing.expectEqual(@as(usize, 0), sp.row % rb);
                try std.testing.expect(sp.local + sp.rows <= m.send_rows);
                try std.testing.expect(sp.gathered + sp.rows <= 2 * m.send_rows);
                for (owners[sp.row .. sp.row + sp.rows]) |*count| count.* += 1;
                var row = sp.row;
                while (row < sp.row + sp.rows) : (row += rb) {
                    try std.testing.expectEqual(@min(rb, rows - row), @min(rb, sp.row + sp.rows - row));
                }
                const src = source[sp.row * kk ..][0 .. sp.rows * kk];
                const dst = gathered[sp.gathered * kk ..][0 .. sp.rows * kk];
                @memcpy(dst, src);
            };
            for (0..2) |rank| for (m.spans(rank)) |sp| {
                @memcpy(merged[sp.row * kk ..][0 .. sp.rows * kk], gathered[sp.gathered * kk ..][0 .. sp.rows * kk]);
            };
            for (owners) |count| try std.testing.expectEqual(@as(u8, 1), count);
            try std.testing.expect(m.send_rows <= rows);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(source), std.mem.sliceAsBytes(merged));
        }
    }
}
