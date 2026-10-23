//! DeepSeek-V4.1's text side on the 1.0 server (M7): the reasoning efforts as the numbers its chat template takes.
//! The checkpoint's template turns "low", "high" and "max" into 50, 75 and 100 and takes any integer 1-100 as it is
//! (the "Reasoning Effort: N" line before the first turn when thinking). Every effort name takes one of those three:
//! minimal and low 50; medium, high and xhigh 75; max and ultra 100. The engine reports these as its Info.efforts, so
//! the server passes each name as asked and the template hears its number. A request without one hears the
//! template's default ("high": 75), the number the lane sends; "none" turns thinking off.
//! Reasoning split ("</think>" closes the block the generation prompt opens) and DSML tool calls are the server's own.

/// How the server shapes chat messages for this template, as engine_api.MessageForm takes it: text parts joined with
/// blank lines, each leading system message its own system turn, a history call's bad arguments under "arguments".
pub const message_form = struct {
    pub const part_separator = "\n\n";
    pub const merge_leading_system = false;
    pub const invalid_arguments_key = "arguments";
};

/// A name and its number, as engine_api.Effort takes them.
pub const Effort = struct { name: []const u8, value: i64 };

pub const efforts = [_]Effort{
    .{ .name = "minimal", .value = 50 },
    .{ .name = "low", .value = 50 },
    .{ .name = "medium", .value = 75 },
    .{ .name = "high", .value = 75 },
    .{ .name = "xhigh", .value = 75 },
    .{ .name = "max", .value = 100 },
    .{ .name = "ultra", .value = 100 },
};

test "every effort name takes the template's 50, 75 or 100" {
    const std = @import("std");
    const want = [_]struct { []const u8, i64 }{ .{ "minimal", 50 }, .{ "low", 50 }, .{ "medium", 75 }, .{ "high", 75 }, .{ "xhigh", 75 }, .{ "max", 100 }, .{ "ultra", 100 } };
    try std.testing.expectEqual(want.len, efforts.len);
    for (want, efforts) |w, e| {
        try std.testing.expectEqualStrings(w[0], e.name);
        try std.testing.expectEqual(w[1], e.value);
    }
}
