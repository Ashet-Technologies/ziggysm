const std = @import("std");

pub fn main() !void {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    if (argv.len != 2)
        @panic("poop");

    const script = try std.fs.cwd().readFileAlloc(allocator, argv[1], 1 << 20);
    defer allocator.free(script);

    //

    var tokenizer: Tokenizer = .{ .data = script };

    while (tokenizer.next()) |token| {
        std.log.info("token({s}) = '{'}'", .{ @tagName(token.type), std.zig.fmtEscapes(token.text) });
    }
}

const Token = struct {
    pub const Type = enum {
        statemachine,
        submachine,
        call,
        yield,
        @"try",
        jump,
        state,
        proc,
        @"if",
        @"while",
        @"return",
        state_ref,
        plain_code,
        @"{",
        @"}",
        @"(",
        @")",
    };

    type: Type,
    text: []const u8,
};

const Tokenizer = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn next(tokenizer: *Tokenizer) ?Token {
        const data = tokenizer.data;
        if (tokenizer.pos >= data.len) {
            return null;
        }

        const start = tokenizer.pos;
        const head = data[tokenizer.pos];

        switch (head) {
            '{' => {
                tokenizer.pos += 1;
                return .{ .type = .@"{", .text = tokenizer.data[start..tokenizer.pos] };
            },
            '}' => {
                tokenizer.pos += 1;
                return .{ .type = .@"}", .text = tokenizer.data[start..tokenizer.pos] };
            },
            '(' => {
                tokenizer.pos += 1;
                return .{ .type = .@"(", .text = tokenizer.data[start..tokenizer.pos] };
            },
            ')' => {
                tokenizer.pos += 1;
                return .{ .type = .@")", .text = tokenizer.data[start..tokenizer.pos] };
            },

            '$' => {
                if (tokenizer.pos + 1 >= data.len) {
                    tokenizer.pos += 1;
                    return .{ .type = .plain_code, .text = tokenizer.data[start..tokenizer.pos] };
                }

                tokenizer.pos += 1;

                while (true) {
                    if (tokenizer.pos >= data.len) {
                        break;
                    }
                    const char = data[tokenizer.pos];
                    switch (char) {
                        'a'...'z' => {},
                        'A'...'Z' => {},
                        '0'...'9' => {},
                        '{', '_', '}' => {},
                        else => break,
                    }
                    tokenizer.pos += 1;
                }

                const text = data[start..tokenizer.pos];
                if (std.mem.eql(u8, text, "$statemachine"))
                    return .{ .type = .statemachine, .text = text };
                if (std.mem.eql(u8, text, "$submachine"))
                    return .{ .type = .submachine, .text = text };
                if (std.mem.eql(u8, text, "$call"))
                    return .{ .type = .call, .text = text };
                if (std.mem.eql(u8, text, "$yield"))
                    return .{ .type = .yield, .text = text };
                if (std.mem.eql(u8, text, "$try"))
                    return .{ .type = .@"try", .text = text };
                if (std.mem.eql(u8, text, "$jump"))
                    return .{ .type = .jump, .text = text };
                if (std.mem.eql(u8, text, "$state"))
                    return .{ .type = .state, .text = text };
                if (std.mem.eql(u8, text, "$proc"))
                    return .{ .type = .proc, .text = text };
                if (std.mem.eql(u8, text, "$if"))
                    return .{ .type = .@"if", .text = text };
                if (std.mem.eql(u8, text, "$while"))
                    return .{ .type = .@"while", .text = text };
                if (std.mem.eql(u8, text, "$return"))
                    return .{ .type = .@"return", .text = text };

                if (std.mem.startsWith(u8, text, "${") and std.mem.endsWith(u8, text, "}"))
                    return .{ .type = .plain_code, .text = text };

                std.log.err("wtf: '{'}'", .{std.zig.fmtEscapes(text)});
                @panic("wtf");
            },

            else => {
                // regular code span...

                var in_string: bool = false;
                var nesting: usize = 0;
                while (true) {
                    if (tokenizer.pos >= data.len) {
                        return .{ .type = .plain_code, .text = data[start..tokenizer.pos] };
                    }
                    const char = data[tokenizer.pos];
                    tokenizer.pos += 1;
                    if (in_string) {
                        switch (char) {
                            '"' => in_string = false,
                            '\\' => tokenizer.pos += 1,
                            else => {},
                        }
                    } else {
                        switch (char) {
                            '$', '{', '}', '(', ')' => if (nesting == 0) { // end of "non-code"
                                tokenizer.pos -= 1;
                                return .{ .type = .plain_code, .text = data[start..tokenizer.pos] };
                            },

                            '[' => nesting += 1,
                            ']' => nesting -= 1,

                            '"' => in_string = true,

                            else => {},
                        }
                    }
                }
            },
        }
    }
};
