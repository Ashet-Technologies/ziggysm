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

    var document = try parse(allocator, script);
    defer document.deinit();

    try ast.dump(document, std.io.getStdOut().writer());
}

pub fn Token(comptime E: type) type {
    return struct {
        pub const Type: type = E;

        type: Type,
        text: []const u8,
    };
}

pub const TopLevelToken = Token(enum {
    // always
    plain_code,

    // toplevel
    statemachine,
    submachine,
    @"async",
    proc,
});

pub fn parse(allocator: std.mem.Allocator, text: []const u8) !ast.Document {
    var tokenizer: Tokenizer = .{ .data = text };

    errdefer |err| {
        var line: usize = 1;
        var col: usize = 1;
        for (tokenizer.data[0..tokenizer.pos]) |c| {
            if (c == '\n') {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }

        std.log.err("{}:{}: error.{s} at {}", .{
            line,
            col,
            @errorName(err),

            ast.TextBlock{ .text = tokenizer.data[tokenizer.pos..] },
        });
    }

    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();

    var parser: Parser = .{
        .tokenizer = &tokenizer,
        .allocator = arena.allocator(),
    };

    var list: std.ArrayList(ast.TopLevelNode) = .init(parser.allocator);
    defer list.deinit();

    while (try parser.accept_top_level_node()) |tln| {
        if (tln == .raw_code and tln.raw_code.text.len == 0)
            continue;
        try list.append(tln);
    }

    const top_level_nodes = try list.toOwnedSlice();
    errdefer arena.allocator().free(top_level_nodes);

    return .{
        .arena = arena,
        .top_level_nodes = top_level_nodes,
    };
}

pub const Parser = struct {
    tokenizer: *Tokenizer,
    allocator: std.mem.Allocator,

    fn accept_top_level_node(parser: *Parser) !?ast.TopLevelNode {
        const token = parser.tokenizer.next_toplevel() orelse return null;

        switch (token.type) {
            .plain_code => return .{ .raw_code = .{ .text = token.text } },

            .@"async" => {
                const name = parser.tokenizer.next_word() orelse return error.SyntaxError;

                const argv = try parser.next_parens_list();

                const return_type = parser.tokenizer.lex_raw_code(.{
                    .nest_braces = true,
                    .nest_parens = true,
                    .only_nesting_group = false,
                    .stop_chars = ";",
                    .include_stop_char = true,
                });

                return .{
                    .async_func = .{
                        .name = name,
                        .arguments = argv,
                        .return_type = .{ .text = return_type },
                    },
                };
            },

            .statemachine => {
                const sm = try parser.accept_state_machine();
                return .{ .state_machine = sm };
            },
            .submachine => {
                const sm = try parser.accept_state_machine();
                return .{ .sub_machine = sm };
            },
            .proc => {
                const sm = try parser.accept_state_machine();
                return .{ .process = sm };
            },
        }
    }

    fn accept_state_machine(parser: *Parser) !ast.StateMachine {
        const name = parser.tokenizer.next_word() orelse return error.SyntaxError;

        const argv = try parser.next_parens_list();

        const return_type = parser.tokenizer.lex_raw_code(.{
            .nest_braces = true,
            .nest_parens = true,
            .only_nesting_group = false,
            .stop_chars = "{",
            .include_stop_char = false,
        });

        const body = try parser.accept_block();

        return .{
            .name = name,
            .arguments = argv,
            .return_type = .{ .text = return_type },
            .body = body,
        };
    }

    fn accept_block(parser: *Parser) error{ OutOfMemory, SyntaxError }!ast.Block {
        var statements: std.ArrayList(ast.Statement) = .init(parser.allocator);
        defer statements.deinit();

        const tok = parser.tokenizer;

        tok.skip_space();
        if (tok.peek() != '{')
            return error.SyntaxError;
        tok.pos += 1;

        while (!tok.eof()) {
            tok.skip_space();
            if (tok.peek() == '}') {
                tok.pos += 1;
                break;
            }

            const stmt = try parser.accept_statement();
            if (stmt == .raw_code and stmt.raw_code.text.len == 0) {
                continue;
            }
            try statements.append(stmt);
        }

        std.debug.assert(tok.data[tok.pos - 1] == '}');

        return .{
            .statements = try statements.toOwnedSlice(),
        };
    }

    fn accept_statement(parser: *Parser) !ast.Statement {
        const tok = parser.tokenizer;
        if (tok.peek() == '$') {
            if (tok.next_word()) |keyword| {
                if (std.mem.eql(u8, keyword, "$call")) {
                    const node = try parser.accept_call_like();
                    return .{ .call = node };
                }
                if (std.mem.eql(u8, keyword, "$yield")) {
                    const node = try parser.accept_call_like();
                    return .{ .yield = node };
                }
                if (std.mem.eql(u8, keyword, "$jump")) {
                    const node = try parser.accept_call_like();
                    return .{ .jump = node };
                }
                if (std.mem.eql(u8, keyword, "$state")) {
                    const name = parser.tokenizer.next_word() orelse return error.SyntaxError;
                    try parser.accept_literal(":");
                    const state_type = parser.tokenizer.lex_raw_code(.{
                        .nest_braces = true,
                        .nest_parens = true,
                        .stop_chars = "=",
                        .include_stop_char = false,
                        .only_nesting_group = false,
                    });
                    try parser.accept_literal("=");
                    const init_value = parser.tokenizer.lex_raw_code(.{
                        .nest_braces = true,
                        .nest_parens = true,
                        .stop_chars = ";",
                        .include_stop_char = false,
                        .only_nesting_group = false,
                    });
                    try parser.accept_literal(";");

                    return .{ .state_variable = .{
                        .variable_name = name,
                        .type_spec = .{ .text = state_type },
                        .initial_value = .{ .text = init_value },
                    } };
                }
                if (std.mem.eql(u8, keyword, "$if")) {
                    const condition = try parser.next_parens_group();

                    const body = try parser.accept_block();

                    return .{ .if_condition = .{
                        .condition = condition,
                        .true_body = body,
                    } };
                }
                if (std.mem.eql(u8, keyword, "$while")) {
                    const condition = try parser.next_parens_group();

                    const body = try parser.accept_block();

                    return .{ .while_loop = .{
                        .condition = condition,
                        .body = body,
                    } };
                }
                if (std.mem.eql(u8, keyword, "$return")) {
                    const value = parser.tokenizer.lex_raw_code(.{
                        .nest_braces = true,
                        .nest_parens = true,
                        .stop_chars = ";",
                        .include_stop_char = false,
                        .only_nesting_group = false,
                    });
                    try parser.accept_literal(";");
                    return .{
                        .@"return" = .{
                            .value = if (value.len > 0) .{ .text = value } else null,
                        },
                    };
                }
                if (std.mem.eql(u8, keyword, "$break")) {
                    return .@"break";
                }
                if (std.mem.eql(u8, keyword, "$continue")) {
                    return .@"continue";
                }

                std.log.err("unknown keyword '{s}'", .{keyword});

                return error.SyntaxError;
            }
        }

        const span = tok.lex_raw_code(.{
            .nest_parens = true,
            .nest_braces = true,
            .only_nesting_group = false,
            .stop_chars = "}",
            .include_stop_char = false,
        });
        if (span.len == 0)
            @panic("ohno");

        return .{
            .raw_code = .{ .text = span },
        };
    }

    fn accept_call_like(parser: *Parser) !ast.CallLike {
        const tok = parser.tokenizer;

        const with_try = parser.try_accept_literal("$try");

        const name = tok.next_word() orelse return error.SyntaxError;
        const argv = try parser.next_parens_list();

        const result_target: ?ast.TextBlock = if (parser.try_accept_literal("->")) blk: {
            const code = tok.lex_raw_code(.{
                .stop_chars = ";",
                .include_stop_char = false,
                .nest_braces = true,
                .nest_parens = true,
                .only_nesting_group = false,
            });
            if (code.len == 0)
                return error.SyntaxError;
            break :blk .{ .text = code };
        } else null;

        try parser.accept_literal(";");

        return .{
            .function_name = name,
            .parameters = argv,
            .output_to = result_target,
            .with_try = with_try,
        };
    }

    fn try_accept_literal(parser: *Parser, text: []const u8) bool {
        parser.tokenizer.skip_space();
        if (!parser.tokenizer.starts_with(text))
            return false;
        parser.tokenizer.pos += text.len;
        return true;
    }
    fn accept_literal(parser: *Parser, text: []const u8) !void {
        parser.tokenizer.skip_space();
        if (!parser.tokenizer.starts_with(text))
            return error.SyntaxError;
        parser.tokenizer.pos += text.len;
    }

    pub fn next_parens_list(parser: *Parser) ![]ast.TextBlock {
        const tokenizer: *Tokenizer = parser.tokenizer;

        try parser.accept_literal("(");

        var list: std.ArrayList(ast.TextBlock) = .init(parser.allocator);
        defer list.deinit();

        while (true) {
            tokenizer.skip_space();
            const group = tokenizer.lex_raw_code(.{
                .nest_braces = true,
                .nest_parens = true,
                .stop_chars = ",)",
                .only_nesting_group = false,
                .include_stop_char = false,
            });

            if (group.len != 0) {
                try list.append(.{ .text = group });
            }

            if (!parser.try_accept_literal(","))
                break;
        }

        try parser.accept_literal(")");

        return try list.toOwnedSlice();
    }

    pub fn next_parens_group(parser: *Parser) !ast.TextBlock {
        const tokenizer: *Tokenizer = parser.tokenizer;

        try parser.accept_literal("(");

        tokenizer.skip_space();
        const group = tokenizer.lex_raw_code(.{
            .nest_braces = true,
            .nest_parens = true,
            .stop_chars = ")",
            .only_nesting_group = false,
            .include_stop_char = false,
        });

        try parser.accept_literal(")");

        return .{ .text = group };
    }
};

const Tokenizer = struct {
    data: []const u8,
    pos: usize = 0,
    nesting: usize = 0,

    fn eof(tok: Tokenizer) bool {
        return tok.pos >= tok.data.len;
    }

    fn peek(tok: Tokenizer) ?u8 {
        return if (tok.eof()) null else tok.data[tok.pos];
    }

    fn starts_with(tok: Tokenizer, text: []const u8) bool {
        return std.mem.startsWith(u8, tok.data[tok.pos..], text);
    }

    pub const whitespace = " \t\r\n";

    fn is_space(chr: u8) bool {
        return switch (chr) {
            ' ', '\t', '\r', '\n' => true,
            else => false,
        };
    }

    fn is_word(chr: u8) bool {
        return switch (chr) {
            'a'...'z' => true,
            'A'...'Z' => true,
            '0'...'9' => true,
            '_', '$' => true,
            else => false,
        };
    }

    fn skip_space(tok: *Tokenizer) void {
        const data = tok.data;
        while (tok.pos < data.len and is_space(data[tok.pos])) {
            tok.pos += 1;
        }
    }

    pub fn next_word(tokenizer: *Tokenizer) ?[]const u8 {
        tokenizer.skip_space();

        const data = tokenizer.data;

        if (tokenizer.pos >= data.len) {
            return null;
        }

        const start = tokenizer.pos;
        while (tokenizer.pos < data.len and is_word(data[tokenizer.pos])) {
            tokenizer.pos += 1;
        }

        if (tokenizer.pos == start) {
            tokenizer.pos = start;
            return null;
        }

        const word = data[start..tokenizer.pos];

        if (std.mem.eql(u8, word, "$")) {
            tokenizer.pos = start;
            return null;
        }

        return word;
    }

    pub fn next_toplevel(tokenizer: *Tokenizer) ?TopLevelToken {
        const data = tokenizer.data;
        if (tokenizer.pos >= data.len) {
            return null;
        }

        const head = data[tokenizer.pos];
        if (head == '$') {
            const keyword = tokenizer.lex_keyword();

            if (std.mem.eql(u8, keyword, "$async"))
                return .{ .type = .@"async", .text = keyword };

            if (std.mem.eql(u8, keyword, "$statemachine"))
                return .{ .type = .statemachine, .text = keyword };

            if (std.mem.eql(u8, keyword, "$submachine"))
                return .{ .type = .submachine, .text = keyword };

            if (std.mem.eql(u8, keyword, "$proc"))
                return .{ .type = .proc, .text = keyword };
        }

        const raw_code = tokenizer.lex_raw_code(.{
            .nest_braces = true,
            .nest_parens = true,
            .only_nesting_group = false,
            .include_stop_char = false,
        });
        return .{
            .type = .plain_code,
            .text = raw_code,
        };
    }

    fn strip_ws(text: []const u8) []const u8 {
        return std.mem.trim(u8, text, Tokenizer.whitespace);
    }

    fn lex_raw_code(tokenizer: *Tokenizer, options: struct {
        nest_braces: bool,
        nest_parens: bool,
        only_nesting_group: bool,
        stop_chars: []const u8 = "",
        include_stop_char: bool,
    }) []const u8 {
        const data = tokenizer.data;
        const start = tokenizer.pos;

        std.debug.assert(std.mem.startsWith(u8, data[start..], "${") or !std.mem.startsWith(u8, data[start..], "$"));

        // regular code span...

        var in_string: bool = false;
        var nesting: usize = 0;
        while (true) {
            if (tokenizer.pos >= data.len) {
                return strip_ws(data[start..tokenizer.pos]);
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
                if (nesting == 0 and options.stop_chars.len > 0 and std.mem.indexOfScalar(u8, options.stop_chars, char) != null) {
                    if (!options.include_stop_char) {
                        tokenizer.pos -= 1;
                    }
                    return strip_ws(data[start..tokenizer.pos]);
                }

                select: switch (char) {
                    '{' => if (options.nest_braces) {
                        nesting += 1;
                    } else {
                        continue :select '$';
                    },

                    '}' => if (options.nest_braces) {
                        nesting -= 1;
                    } else {
                        continue :select '$';
                    },

                    '(' => if (options.nest_parens) {
                        nesting += 1;
                    } else {
                        continue :select '$';
                    },

                    ')' => if (options.nest_parens) {
                        nesting -= 1;
                    } else {
                        continue :select '$';
                    },

                    // end of "non-code", but only on non-nested setups and not on "${"
                    '$' => if (nesting == 0 and (tokenizer.pos >= data.len or data[tokenizer.pos] != '{')) {
                        tokenizer.pos -= 1;
                        return strip_ws(data[start..tokenizer.pos]);
                    },

                    '[' => nesting += 1,
                    ']' => nesting -= 1,

                    '"' => in_string = true,

                    else => {},
                }
                if (options.only_nesting_group and nesting == 0) {
                    return strip_ws(data[start..tokenizer.pos]);
                }
            }
        }
    }

    fn lex_keyword(tokenizer: *Tokenizer) []const u8 {
        const data = tokenizer.data;
        const start = tokenizer.pos;

        std.debug.assert(data[start] == '$');
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

        return data[start..tokenizer.pos];
    }
};

pub const ast = struct {
    pub const Document = struct {
        arena: std.heap.ArenaAllocator,
        top_level_nodes: []TopLevelNode,

        pub fn deinit(doc: *Document) void {
            doc.arena.deinit();
            doc.* = undefined;
        }
    };

    pub const TopLevelNode = union(enum) {
        raw_code: TextBlock,
        state_machine: StateMachine,
        sub_machine: StateMachine,
        process: StateMachine,
        async_func: AsyncFunction,
    };

    pub const AsyncFunction = struct {
        name: []const u8,
        arguments: []const TextBlock,
        return_type: TextBlock,
    };

    pub const StateMachine = struct {
        name: []const u8,
        arguments: []const TextBlock,
        return_type: TextBlock,
        body: Block,
    };

    pub const Block = struct {
        statements: []const Statement,
    };

    pub const Statement = union(enum) {
        raw_code: TextBlock,
        state_variable: StateVariable,
        yield: CallLike,
        call: CallLike,
        jump: CallLike,
        @"return": Return,
        while_loop: WhileLoop,
        if_condition: IfCondition,
        @"break",
        @"continue",
    };

    pub const StateVariable = struct {
        variable_name: []const u8,
        type_spec: TextBlock,
        initial_value: TextBlock,
    };

    pub const CallLike = struct {
        with_try: bool,
        output_to: ?TextBlock,
        function_name: []const u8,
        parameters: []const TextBlock,
    };

    pub const Jump = struct {
        jump_to: []const u8,
        parameters: []const TextBlock,
    };

    pub const Return = struct {
        value: ?TextBlock,
    };

    pub const WhileLoop = struct {
        condition: TextBlock,
        body: Block,
    };

    pub const IfCondition = struct {
        condition: TextBlock,
        true_body: Block,
    };

    pub const TextBlock = struct {
        text: []const u8,

        pub fn format(tb: TextBlock, fmt: []const u8, opt: std.fmt.FormatOptions, writer: anytype) !void {
            const max_len = 32;
            const split_len = @divExact(max_len, 2);
            _ = fmt;
            _ = opt;
            try writer.writeAll("TextBlock('");
            if (tb.text.len < max_len) {
                try writer.print("{'}", .{std.zig.fmtEscapes(tb.text)});
            } else {
                const head = tb.text[0..split_len];
                const tail = tb.text[tb.text.len - split_len ..];
                try writer.print("{'}…{'}", .{
                    std.zig.fmtEscapes(head),
                    std.zig.fmtEscapes(tail),
                });
            }
            try writer.writeAll("')");
        }
    };

    pub fn dump(doc: Document, writer: anytype) !void {
        const Dumper = struct {
            const Dumper = @This();

            stream: @TypeOf(writer),

            fn render(dmp: Dumper, d: Document) !void {
                for (d.top_level_nodes) |node| {
                    switch (node) {
                        .raw_code => |code| try dmp.stream.print("{}\n", .{code}),

                        .async_func => |func| {
                            try dmp.stream.print("async {s}({any}) {}\n", .{
                                func.name,
                                func.arguments,
                                func.return_type,
                            });
                        },

                        .process, .state_machine, .sub_machine => |sm| {
                            try dmp.stream.print("{s} {s}({any}) {}\n", .{ @tagName(node), sm.name, sm.arguments, sm.return_type });
                            try dmp.block(sm.body, 0);
                        },
                    }
                }
            }

            fn block(dmp: Dumper, blk: Block, depth: usize) @TypeOf(writer).Error!void {
                const prefix = ("\t" ** 32)[0..depth];

                try dmp.stream.print("{s}{{\n", .{prefix});
                for (blk.statements) |stmt| {
                    try dmp.statement(stmt, depth + 1);
                }
                try dmp.stream.print("{s}}}\n", .{prefix});
            }
            fn statement(dmp: Dumper, stmt: Statement, depth: usize) !void {
                const prefix = ("\t" ** 32)[0..depth];

                try dmp.stream.writeAll(prefix);
                switch (stmt) {
                    .raw_code => |code| try dmp.stream.print("{}\n", .{code}),
                    .state_variable => |state| {
                        try dmp.stream.print("$state {s}: {} = {};\n", .{
                            state.variable_name,
                            state.type_spec,
                            state.initial_value,
                        });
                    },
                    .yield, .call, .jump => |call| {
                        try dmp.stream.print("{s}{s} {s}({any})", .{
                            @tagName(stmt),
                            if (call.with_try) " $try" else "",
                            call.function_name,
                            call.parameters,
                        });
                        if (call.output_to) |output| {
                            try dmp.stream.print("-> {}", .{output});
                        }
                        try dmp.stream.writeAll(";\n");
                    },
                    .@"return" => |ret| {
                        try dmp.stream.writeAll("$return");
                        if (ret.value) |value| {
                            try dmp.stream.print(" {}", .{value});
                        }

                        try dmp.stream.writeAll(";\n");
                    },
                    .while_loop => |cond| {
                        try dmp.stream.writeAll("$while_loop(");
                        try dmp.stream.print("{}", .{cond.condition});
                        try dmp.stream.writeAll(")\n");
                        try dmp.block(cond.body, depth);
                    },
                    .if_condition => |cond| {
                        try dmp.stream.writeAll("$if_condition(");
                        try dmp.stream.print("{}", .{cond.condition});
                        try dmp.stream.writeAll(")\n");
                        try dmp.block(cond.true_body, depth);
                    },
                    .@"break" => try dmp.stream.writeAll("$break;\n"),
                    .@"continue" => try dmp.stream.writeAll("$continue;\n"),
                }
            }
        };

        var dumper: Dumper = .{ .stream = writer };

        try dumper.render(doc);
    }
};
