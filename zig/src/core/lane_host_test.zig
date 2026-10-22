//! Host memory reporting and isolated request refusal through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

const PrefillProbe = struct {
    mutex: std.Io.Mutex = .init,
    engine: api.Engine,
    host: *LaneHost,
    requests: [2]Request,
    tokens: [3]std.ArrayList(u32) = @splat(.empty),
    done: [3]?Reason = @splat(null),
    prefilled: [3]bool = @splat(false),
    chunks: [3]usize = @splat(0),
    last_tokens: [3]usize = @splat(0),
    submitted: bool = false,
    bad_order: bool = false,
    cancel: bool = false,
    cancelled: bool = false,
    limited: bool = false,
    closing: bool = false,
    refusals: usize = 0,

    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const p: *@This() = @ptrCast(@alignCast(ctx));
        p.mutex.lockUncancelable(std.testing.io);
        defer p.mutex.unlock(std.testing.io);
        const i: usize = @intCast(id - 1);
        switch (e.*) {
            .prefilled => p.prefilled[i] = true,
            .tokens => |tokens| {
                if (!p.prefilled[i]) p.bad_order = true;
                p.tokens[i].appendSlice(std.testing.allocator, tokens) catch {
                    p.bad_order = true;
                };
                if (id == 1 and !p.submitted) {
                    p.submitted = true;
                    for (&p.requests, 2..) |*r, next_id| p.engine.submit(next_id, r, .{ .ctx = p, .event = event }) catch {
                        p.bad_order = true;
                    };
                }
                if (id == 1 and p.cancel and !p.cancelled and p.chunks[1] == 1) {
                    p.cancelled = true;
                    p.engine.cancel(2);
                }
            },
            .finished => |f| p.done[i] = f.reason,
        }
    }

    fn chunk(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const p: *@This() = @ptrCast(@alignCast(ctx));
        p.mutex.lockUncancelable(std.testing.io);
        const i: usize = @intCast(s.prompt()[0] - 1);
        if (i > 0 and p.chunks[i] > 0 and p.done[0] == null and p.tokens[0].items.len <= p.last_tokens[i]) p.bad_order = true;
        if (i == 2 and !p.prefilled[1] and p.done[1] == null) p.bad_order = true;
        p.last_tokens[i] = p.tokens[0].items.len;
        p.chunks[i] += 1;
        p.mutex.unlock(std.testing.io);
        if (p.closing and i == 1) {
            while (true) {
                p.host.mutex.lockUncancelable(std.testing.io);
                const closing = p.host.closing;
                p.host.mutex.unlock(std.testing.io);
                if (closing) break;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    }

    fn part(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const x: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const p: *@This() = @ptrCast(@alignCast(x.prefill_hook_ctx.?));
        if (p.limited and x.lanes.count() > 0 and !x.lanes.contains(s)) {
            p.refusals += 1;
            return error.ContextFull;
        }
        return x.backend().prefillPart(s);
    }

    fn wait(p: *@This()) void {
        while (true) {
            p.mutex.lockUncancelable(std.testing.io);
            const done = if (p.closing) p.chunks[1] > 0 else p.done[0] != null and p.done[1] != null and p.done[2] != null;
            p.mutex.unlock(std.testing.io);
            if (done) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

const PrefillMode = enum { interleave, cancel, limited, closing };

fn runPrefill(mode: PrefillMode) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .interleave = true, .prefill_chunks = 6 };
    defer target.deinit();
    var vtable = target.backend().vtable.*;
    vtable.prefill_part = PrefillProbe.part;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &vtable }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    const first: Request = .{ .prompt = &.{ 1, 4, 1 }, .max_tokens = 32, .drafts = false };
    var probe: PrefillProbe = .{
        .engine = host.engine(),
        .host = &host,
        .requests = .{
            .{ .prompt = &.{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 }, .max_tokens = 8, .drafts = false, .sampling = .{ .seed = 7, .temperature = 0.7 } },
            .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 8, .drafts = false },
        },
        .cancel = mode == .cancel,
        .limited = mode == .limited,
        .closing = mode == .closing,
    };
    defer for (&probe.tokens) |*tokens| tokens.deinit(gpa);
    target.prefill_hook = PrefillProbe.chunk;
    target.prefill_hook_ctx = &probe;
    {
        try host.start();
        defer host.stop();
        try probe.engine.submit(1, &first, .{ .ctx = &probe, .event = PrefillProbe.event });
        probe.wait();
    }
    try std.testing.expect(!probe.bad_order);
    try std.testing.expectEqual(@as(usize, if (probe.cancel or probe.closing) 1 else 6), probe.chunks[1]);
    try std.testing.expectEqual(@as(usize, if (probe.closing) 0 else 6), probe.chunks[2]);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    try std.testing.expect(host.filling == null);
    if (probe.closing) {
        for (probe.done) |done| try std.testing.expectEqual(Reason.cancelled, done.?);
        try std.testing.expectEqual(@as(usize, 0), core.activeCount());
        return;
    }
    if (probe.limited) try std.testing.expect(probe.refusals > 0);
    const requests = [_]Request{ first, probe.requests[0], probe.requests[1] };
    for (requests, probe.tokens, probe.done, 0..) |r, tokens, done, i| {
        if (probe.cancel and i == 1) {
            try std.testing.expectEqual(Reason.cancelled, done.?);
            try std.testing.expectEqual(@as(usize, 0), tokens.items.len);
            continue;
        }
        try std.testing.expectEqual(Reason.length, done.?);
        try std.testing.expectEqual(@as(usize, r.max_tokens), tokens.items.len);
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, r.prompt);
        for (tokens.items) |token| {
            try std.testing.expectEqual(lanes.fake.next(history.items, r.sampling, history.items.len), token);
            try history.append(gpa, token);
        }
    }
}

test "a lane host runs a round between prompt chunks and fills queued jobs one at a time" {
    try runPrefill(.interleave);
}

test "a lane host releases a cancelled partial prompt and continues its running and queued streams" {
    try runPrefill(.cancel);
}

test "a lane host requeues a stepped prompt until the backend has context room" {
    try runPrefill(.limited);
}

test "a closing lane host releases its running, filling and queued jobs" {
    try runPrefill(.closing);
}
