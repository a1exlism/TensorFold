//! A streamed context-window refusal is an error event, and every other refusal is a 400 before the stream opens.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic_translate = @import("anthropic_translate.zig");
const responses_translate = @import("responses_translate.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt" (6 tokens).
const TestText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An 8-token window whose engine answers "hi" at once.
const HiEngine = struct {
    submitted: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *HiEngine = @ptrCast(@alignCast(ctx));
        e.submitted += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// What the route's output saw: a whole reply's status, error code and sampling label, or a stream and its first role.
const Seen = struct {
    gone_at_open: bool = false, // the client is gone when the stream opens
    opened: bool = false,
    events: usize = 0,
    role_first: bool = false,
    status: ?u16 = null,
    context_code: bool = false,
    label: ?enum { exact, greedy } = null,
    preparing: i64 = 0,
    submitted: usize = 0,
    top_k_refusal: bool = false,

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        if (c.gone_at_open) return error.Closed;
        c.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.events += 1;
        const p = payload orelse return;
        if (p.get("error")) |err| {
            const code = err.get("code") orelse return;
            c.context_code = code == .string and std.mem.eql(u8, code.string, "context_length_exceeded");
            return;
        }
        if (c.events != 1) return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        const role = (choices.array[0].get("delta") orelse return).get("role") orelse return;
        c.role_first = role == .string and std.mem.eql(u8, role.string, "assistant");
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        if (payload.get("tensorfold")) |t| if (t.get("sampling")) |l| if (l == .string) {
            c.label = if (std.mem.eql(u8, l.string, "exact")) .exact else if (std.mem.eql(u8, l.string, "greedy")) .greedy else null;
        };
        const err = payload.get("error") orelse return;
        if (err.get("message")) |m| if (m == .string) {
            c.top_k_refusal = std.mem.eql(u8, m.string, "top_k must be between 1 and 56 when temperature > 0");
        };
        const code = err.get("code") orelse return;
        c.context_code = code == .string and std.mem.eql(u8, code.string, "context_length_exceeded");
    }
};

const Route = enum { chat, messages, responses };

/// A Responses request with no stored conversation to read.
const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

/// `body` through `route`'s translation and the chat handler, as the route runs it; the preparing count after it.
fn send(route: Route, body: []const u8) !Seen {
    return sendTo(route, body, .{});
}

fn sendTo(route: Route, body: []const u8, start: Seen) !Seen {
    return sendWith(route, body, start, null, null);
}

fn sendWith(route: Route, body: []const u8, start: Seen, max_top_k: ?u32, defaults: ?[]const u8) !Seen {
    var text: TestText = .{};
    var backend: HiEngine = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    srv.info.max_top_k = max_top_k;
    if (defaults) |d| srv.config.default_sampling = (try json.parse(a, d)).ok;
    const raw = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    const chat = switch (route) {
        .chat => raw,
        .messages => try anthropic_translate.translate(&cx, raw, false),
        .responses => (try responses_translate.translate(&cx, raw, NoHistory{})).chat,
    };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen = start;
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, chat);
    seen.preparing = srv.preparing.load(.acquire);
    seen.submitted = backend.submitted;
    return seen;
}

fn expectContextInStream(seen: Seen, role: bool) !void {
    try std.testing.expect(seen.opened);
    try std.testing.expectEqual(role, seen.role_first);
    try std.testing.expect(seen.context_code);
    try std.testing.expectEqual(@as(?u16, null), seen.status);
    try std.testing.expect(seen.events >= 2);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a streamed chat over the window reports context_length_exceeded after the role chunk" {
    try expectContextInStream(try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}"), true);
}

test "a streamed chat with tools over the window reports the context error with no role chunk" {
    try expectContextInStream(try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100," ++
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]}"), false);
}

test "a streamed Messages request over the window reports the context error in the stream" {
    try expectContextInStream(try send(.messages, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":100}"), true);
}

test "a streamed Responses request over the window reports the context error in the stream" {
    try expectContextInStream(try send(.responses, "{\"model\":\"test-model\",\"input\":\"x\",\"stream\":true,\"max_output_tokens\":100}"), true);
}

test "a valid streamed chat still opens with the role chunk and gives the preparing count back" {
    const seen = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}");
    try std.testing.expect(seen.opened);
    try std.testing.expect(seen.role_first);
    try std.testing.expect(seen.events >= 3); // the role, the text, the last chunk
    try std.testing.expectEqual(@as(?u16, null), seen.status);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "a stream whose client is gone when it opens gives the preparing count back" {
    const body = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":2}";
    const seen = try sendTo(.chat, body, .{ .gone_at_open = true });
    try std.testing.expect(!seen.opened);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "top_k limits refuse positive-temperature requests before opening or submitting" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]bool{ false, true }) |stream| {
        for ([_]f64{ 0, 1 }) |temp| for ([_]i32{ -1, 0, 20, 56, 57 }) |k| {
            const input = if (route == .responses) "\"input\":\"x\"" else "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]";
            const limit = if (route == .responses) "max_output_tokens" else "max_tokens";
            const body = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",{s},\"{s}\":2,\"temperature\":{d},\"top_k\":{d},\"stream\":{s}}}", .{ input, limit, temp, k, if (stream) "true" else "false" });
            const seen = try sendWith(route, body, .{}, 56, null);
            const refused = temp > 0 and (k < 1 or k > 56);
            try std.testing.expectEqual(refused, seen.top_k_refusal);
            try std.testing.expectEqual(@as(i64, 0), seen.preparing);
            if (refused) {
                try std.testing.expectEqual(@as(?u16, 400), seen.status);
                try std.testing.expect(!seen.opened);
                try std.testing.expectEqual(@as(usize, 0), seen.events);
                try std.testing.expectEqual(@as(usize, 0), seen.submitted);
            } else {
                try std.testing.expectEqual(@as(usize, 1), seen.submitted);
                try std.testing.expectEqual(stream, seen.opened);
                if (!stream) try std.testing.expectEqual(@as(?u16, 200), seen.status);
            }
        };
    };
}

test "top_k limits use the effective sampling after defaults and request overrides" {
    const cases = [_]struct { defaults: []const u8, fields: []const u8, refused: bool }{
        .{ .defaults = "{\"temperature\":1,\"top_k\":20}", .fields = "\"top_k\":0", .refused = true },
        .{ .defaults = "{\"temperature\":1,\"top_k\":0}", .fields = "\"temperature\":0", .refused = false },
        .{ .defaults = "{\"temperature\":0,\"top_k\":0}", .fields = "\"temperature\":1,\"top_k\":20", .refused = false },
        .{ .defaults = "{\"temperature\":1,\"top_k\":57}", .fields = "\"seed\":1", .refused = true },
        .{ .defaults = "{\"temperature\":1,\"top_k\":57}", .fields = "\"top_k\":56", .refused = false },
        .{ .defaults = "{\"temperature\":1,\"top_k\":20}", .fields = "\"top_k\":null,\"temperature\":null", .refused = false },
        .{ .defaults = "{\"temperature\":1,\"top_k\":0}", .fields = "\"top_k\":null,\"temperature\":null", .refused = true },
        .{ .defaults = "{\"temperature\":-1,\"top_k\":57}", .fields = "\"seed\":1", .refused = false },
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        const body = try std.fmt.allocPrint(arena.allocator(), "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"max_tokens\":2,\"stream\":true,{s}}}", .{c.fields});
        const seen = try sendWith(.chat, body, .{}, 56, c.defaults);
        try std.testing.expectEqual(c.refused, seen.top_k_refusal);
        try std.testing.expectEqual(!c.refused, seen.opened);
        try std.testing.expectEqual(@as(usize, if (c.refused) 0 else 1), seen.submitted);
    }
}

test "engines without a top_k limit keep their sampling behavior" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]i32{ -1, 0, 20, 56, 57 }) |k| {
        const body = try std.fmt.allocPrint(arena.allocator(), "{{\"model\":\"test-model\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"max_tokens\":2,\"temperature\":1,\"top_k\":{d}}}", .{k});
        const seen = try send(.chat, body);
        try std.testing.expectEqual(@as(?u16, 200), seen.status);
        try std.testing.expectEqual(@as(usize, 1), seen.submitted);
        try std.testing.expectEqual(.exact, seen.label.?);
    }
}

test "a reply labels a greedy request greedy and a sampled one exact" {
    const greedy = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0}");
    try std.testing.expectEqual(@as(?u16, 200), greedy.status);
    try std.testing.expectEqual(.greedy, greedy.label.?);
    const sampled = try send(.chat, "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2,\"temperature\":0.7,\"seed\":1}");
    try std.testing.expectEqual(.exact, sampled.label.?);
    try std.testing.expectEqual(@as(i64, 0), sampled.preparing);
}
