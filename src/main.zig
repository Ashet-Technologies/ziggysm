const std = @import("std");
const args_parser = @import("args");

const CliArgs = struct {
    help: bool = false,
    output: []const u8 = "",

    pub const shorthands = .{
        .h = "help",
        .o = "output",
    };
};

pub fn main() !u8 {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();

    var cli = args_parser.parseForCurrentProcess(CliArgs, allocator, .print) catch return 1;
    defer cli.deinit();

    if (cli.positionals.len != 1) {
        @panic("poop");
    }

    const file_name = cli.positionals[0];

    const script = try std.fs.cwd().readFileAlloc(allocator, file_name, 1 << 20);
    defer allocator.free(script);

    var document = try parse(allocator, script);
    defer document.deinit();

    try ast.dump(document, std.io.getStdOut().writer());

    var program = try generate_code(allocator, document);
    defer program.deinit();
    {
        const writer = std.io.getStdOut().writer();
        for (program.structure) |node| {
            switch (node) {
                .state_machine => |sm| {
                    try writer.print("StateMachine {s}:\n", .{sm.name});
                    try ir.dump(sm, writer);
                },
                .top_level_code => |code| try writer.print("TopLevel Code: {}\n", .{code}),
            }
        }
    }

    const unformatted_code = blk: {
        var sink: std.ArrayList(u8) = .init(allocator);
        defer sink.deinit();

        for (program.structure) |node| {
            switch (node) {
                .state_machine => |sm| try render(allocator, program, sm, sink.writer()),
                .top_level_code => |code| try sink.appendSlice(code.text), // no processing of global replacements
            }
        }

        break :blk try sink.toOwnedSliceSentinel(0);
    };

    const formatted_code = blk: {
        const zig_ast = try std.zig.Ast.parse(allocator, unformatted_code, .zig);
        if (zig_ast.errors.len > 0)
            break :blk unformatted_code;

        break :blk try zig_ast.render(allocator);
    };

    if (cli.options.output.len > 0) {
        try std.fs.cwd().writeFile(.{
            .sub_path = cli.options.output,
            .data = formatted_code,
        });
    } else {
        try std.io.getStdOut().writeAll(formatted_code);
    }

    return 0;
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

                const argv = try parser.next_parameter_list();

                const return_type = parser.tokenizer.lex_raw_code(.{
                    .nest_braces = true,
                    .nest_parens = true,
                    .only_nesting_group = false,
                    .stop_chars = ";",
                    .include_stop_char = false,
                });

                try parser.accept_literal(";");

                return .{
                    .async_func = .{
                        .name = name,
                        .parameters = argv,
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

        const argv = try parser.next_parameter_list();

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
            .parameters = argv,
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

        var as_state: bool = false;
        const result_target: ?ast.TextBlock = if (parser.try_accept_literal("->")) blk: {
            as_state = parser.try_accept_literal("$state");

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
            .arguments = argv,
            .output_to = result_target,
            .with_try = with_try,
            .as_state = as_state,
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

    pub fn next_parameter_list(parser: *Parser) ![]ast.Parameter {
        const tokenizer: *Tokenizer = parser.tokenizer;

        try parser.accept_literal("(");

        var list: std.ArrayList(ast.Parameter) = .init(parser.allocator);
        defer list.deinit();

        while (true) {
            tokenizer.skip_space();

            if (tokenizer.peek() == ')')
                break;

            const name = tokenizer.next_word() orelse return error.MissingParameterName;

            try parser.accept_literal(":");

            const type_name = tokenizer.lex_raw_code(.{
                .nest_braces = true,
                .nest_parens = true,
                .stop_chars = ",)",
                .only_nesting_group = false,
                .include_stop_char = false,
            });

            try list.append(.{
                .name = name,
                .type = .{ .text = type_name },
            });

            if (!parser.try_accept_literal(","))
                break;
        }

        try parser.accept_literal(")");

        return try list.toOwnedSlice();
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

    pub const Parameter = struct {
        name: []const u8,
        type: TextBlock,
    };

    pub const AsyncFunction = struct {
        name: []const u8,
        parameters: []const Parameter,
        return_type: TextBlock,
    };

    pub const StateMachine = struct {
        name: []const u8,
        parameters: []const Parameter,
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
        as_state: bool,
        output_to: ?TextBlock,
        function_name: []const u8,
        arguments: []const TextBlock,
    };

    pub const Jump = struct {
        jump_to: []const u8,
        arguments: []const TextBlock,
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
                                func.parameters,
                                func.return_type,
                            });
                        },

                        .process, .state_machine, .sub_machine => |sm| {
                            try dmp.stream.print("{s} {s}({any}) {}\n", .{ @tagName(node), sm.name, sm.parameters, sm.return_type });
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
                            call.arguments,
                        });
                        if (call.output_to) |output| {
                            try dmp.stream.print("-> {s}{}", .{
                                if (call.with_try) " $state" else "",
                                output,
                            });
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

const ir = struct {
    pub const TextBlock = ast.TextBlock;

    pub const Label = enum(usize) {
        _,

        pub fn format(lbl: Label, fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;

            try writer.print("lbl{}", .{@intFromEnum(lbl)});
        }
    };

    pub const Program = struct {
        arena: std.heap.ArenaAllocator,

        structure: []Node,
        suspenders: std.StringArrayHashMap(Suspender),

        pub fn deinit(pgm: *Program) void {
            pgm.arena.deinit();
            pgm.* = undefined;
        }
    };

    pub const Node = union(enum) {
        state_machine: StateMachine,
        top_level_code: TextBlock,
    };

    pub const Suspender = struct {
        name: []const u8,
        parameters: []const Parameter,
        return_type: TextBlock,

        pub const Parameter = ast.Parameter;
    };

    pub const StateMachine = struct {
        name: []const u8,

        labels: std.AutoArrayHashMap(Label, LabelData), // id => label info
        // states: std.StringArrayHashMap(TextBlock), // name => type
        instructions: []Instruction,
    };

    pub const LabelData = struct {
        tag: ?[]const u8,
        offset: ?usize,
    };

    pub const SuspendState = struct {
        parameters: []const TextBlock,
        result: ?TextBlock,
    };

    pub const Instruction = union(enum) {
        stop,
        execute: TextBlock,

        branch: Branch,
        true_branch: ConditionalBranch,
        false_branch: ConditionalBranch,
        dyn_branch: DynamicBranch,

        call: CallDynamicTarget,

        yield: Yield,

        set_state: SetState,
    };

    pub const Yield = struct {
        function: []const u8,
        arguments: []const TextBlock,
        output: ?TextBlock,
        use_try: bool,
    };
    pub const Branch = struct {
        target: Label,
    };

    pub const DynamicBranch = struct {
        target: []const u8,
    };

    pub const CallDynamicTarget = struct {
        target: []const u8,
    };

    pub const ConditionalBranch = struct {
        target: Label,
        condition: TextBlock,
    };

    pub const SetState = struct {
        variable: []const u8,
        value: TextBlock,
    };

    pub fn dump(pgm: StateMachine, writer: anytype) !void {
        for (pgm.instructions, 0..) |instr, offset| {
            for (pgm.labels.keys(), pgm.labels.values()) |lbl, data| {
                if (data.offset != offset)
                    continue;
                try writer.print("        {}:\n", .{
                    lbl,
                });
            }
            try writer.print("0x{X:0>4}    ", .{
                offset,
            });
            switch (instr) {
                .stop => try writer.writeAll("STOP"),
                .execute => |code| {
                    try writer.print("EXEC  {}", .{code});
                },
                .yield => |yield| {
                    try writer.print("YIELD {s}, {any}, {s}{?}", .{
                        yield.function,
                        yield.arguments,
                        if (yield.use_try) "try " else "",
                        yield.output,
                    });
                },
                .set_state => |state| {
                    try writer.print("SETST {s}, {}", .{ state.variable, state.value });
                },
                .branch => |branch| {
                    try writer.print("BR    {}", .{branch.target});
                },
                .dyn_branch => |branch| {
                    try writer.print("BR.D  {s}", .{branch.target});
                },
                .call => |branch| {
                    try writer.print("CALL  {s}", .{branch.target});
                },
                .true_branch => |branch| {
                    try writer.print("BR.T  {}, {}", .{ branch.target, branch.condition });
                },
                .false_branch => |branch| {
                    try writer.print("BR.F  {}, {}", .{ branch.target, branch.condition });
                },
            }
            try writer.writeAll("\n");
        }
    }
};

fn generate_code(allocator: std.mem.Allocator, document: ast.Document) !ir.Program {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();

    var structure: std.ArrayList(ir.Node) = .init(arena.allocator());

    var suspenders: std.StringArrayHashMap(ir.Suspender) = .init(arena.allocator());

    // TODO: Create an index for all process, asyncs and submachines:
    for (document.top_level_nodes) |node| {
        switch (node) {
            .state_machine, .raw_code => continue,

            .async_func => |suspender| {
                const previous = try suspenders.fetchPut(suspender.name, .{
                    .name = suspender.name,
                    .parameters = suspender.parameters,
                    .return_type = suspender.return_type,
                });
                if (previous != null)
                    return error.DuplicateSuspender;
            },

            .process => {},
            .sub_machine => {},
        }
    }

    for (document.top_level_nodes) |node| {
        const sm = switch (node) {
            .state_machine => |*sm| sm,
            .raw_code => |code| {
                try structure.append(.{ .top_level_code = code });
                continue;
            },

            .async_func, .process, .sub_machine => continue,
        };

        var cg: CodeGen = .{
            .allocator = arena.allocator(),
            .instructions = .init(arena.allocator()),
            .labels = .init(arena.allocator()),
            .tagged_labels = .init(arena.allocator()),
            .suspenders = &suspenders,
        };

        _ = try cg.generate(sm.*, true);

        // TODO: Generate all dependencies as well here:

        const instructions = try cg.instructions.toOwnedSlice();
        errdefer cg.allocator.free(instructions);

        var labels: std.AutoArrayHashMap(ir.Label, ir.LabelData) = .init(arena.allocator());
        errdefer labels.deinit();

        for (cg.labels.items) |label| {
            try labels.put(label.id, .{
                .offset = label.offset,
                .tag = null,
            });
        }

        try structure.append(.{ .state_machine = .{
            .name = sm.name,
            .instructions = instructions,
            .labels = labels,
        } });
    }

    return .{
        .arena = arena,
        .structure = try structure.toOwnedSlice(),
        .suspenders = suspenders,
    };
}

const CodeGen = struct {
    const Label = struct {
        offset: ?usize = null,
        used: bool = false,
        id: ir.Label,
    };

    allocator: std.mem.Allocator,
    suspenders: *std.StringArrayHashMap(ir.Suspender),

    instructions: std.ArrayList(ir.Instruction),

    labels: std.ArrayList(*Label),
    tagged_labels: std.StringArrayHashMap(?*Label),

    fn create_label(cg: *CodeGen, mode: enum { here, undefined }) !*Label {
        const lbl = try cg.allocator.create(Label);
        errdefer cg.allocator.destroy(lbl);

        lbl.* = .{
            .id = @enumFromInt(cg.labels.items.len),
            .offset = switch (mode) {
                .here => cg.instructions.items.len,
                .undefined => null,
            },
        };
        try cg.labels.append(lbl);

        return lbl;
    }

    fn define_label(cg: *CodeGen, lbl: *Label) !void {
        if (lbl.offset != null)
            @panic("label was defined twice!");
        lbl.offset = cg.instructions.items.len;
    }

    fn tag_label(cg: *CodeGen, lbl: *Label, tag: []const u8) !void {
        const gop = try cg.tagged_labels.getOrPut(tag);
        if (gop.found_existing) {
            if (gop.value_ptr.* != null)
                return error.DuplicateLabelTag;
        }
        gop.value_ptr.* = lbl;
    }

    fn get_tagged_label(cg: *CodeGen, tag: []const u8) !*Label {
        const gop = try cg.tagged_labels.getOrPut(tag);
        if (gop.found_existing) {
            return gop.value_ptr.*.?;
        }
        errdefer _ = cg.tagged_labels.swapRemove(tag);
        gop.value_ptr.* = try cg.create_label(.undefined);
        return gop.value_ptr.*.?;
    }

    fn generate(cg: *CodeGen, sm: ast.StateMachine, is_toplevel: bool) !ir.Label {
        const sm_label = try cg.create_label(.here);

        try cg.tag_label(sm_label, sm.name);

        try cg.generate_block(sm.body, .{
            .scope = sm.name,
            .is_toplevel = is_toplevel,
        });

        try cg.emit(.stop);

        return sm_label.id;
    }

    const BlockContext = struct {
        const Loop = struct {
            break_point: *Label,
            cont_point: *Label,
        };

        scope: []const u8,
        loop: ?Loop = null,
        is_toplevel: bool,
    };

    fn generate_block(cg: *CodeGen, block: ast.Block, ctx: BlockContext) error{ OutOfMemory, SyntaxError, BreakOutsideLoop, ValueReturnOnToplevel, UnknownSuspender, ParameterMismatch }!void {
        for (block.statements) |stmt| {
            try cg.generate_stmt(stmt, ctx);
        }
    }

    fn generate_stmt(cg: *CodeGen, stmt: ast.Statement, ctx: BlockContext) !void {
        switch (stmt) {
            .raw_code => |code| try cg.emit(.{ .execute = code }),

            .state_variable => |state| {
                try cg.emit(.{ .set_state = .{
                    .variable = state.variable_name,
                    .value = state.initial_value,
                } });
            },

            .yield => |yield| {
                const suspender = cg.suspenders.get(yield.function_name) orelse return error.UnknownSuspender;

                if (yield.arguments.len != suspender.parameters.len)
                    return error.ParameterMismatch;

                try cg.emit(.{ .yield = .{
                    .function = yield.function_name,
                    .arguments = yield.arguments,
                    .output = if (yield.output_to) |output_to|
                        if (yield.as_state)
                            .{ .text = try std.fmt.allocPrint(cg.allocator, "${{{s}}}", .{output_to.text}) }
                        else
                            output_to
                    else
                        null,
                    .use_try = yield.with_try,
                } });
            },

            .call => |call| {
                for (call.arguments) |param| {
                    // TODO: Emit call
                    _ = param;
                }

                try cg.emit(.{ .call = .{
                    .target = call.function_name,
                } });
            },

            .@"return" => |ret| {
                if (ctx.is_toplevel) {
                    if (ret.value != null) {
                        return error.ValueReturnOnToplevel;
                    }
                    try cg.emit(.stop);
                } else {
                    if (ret.value) |value| {
                        try cg.emit(.{ .set_state = .{
                            .variable = try cg.get_return_value(ctx.scope),
                            .value = value,
                        } });
                    }
                    try cg.emit(.{ .dyn_branch = .{
                        .target = ctx.scope,
                    } });
                }
            },

            .jump => |jmp| {

                // TODO: Emit state changes here

                if (jmp.with_try)
                    return error.SyntaxError;
                const target = try cg.get_tagged_label(jmp.function_name);
                target.used = true;
                try cg.emit(.{
                    .branch = .{ .target = target.id },
                });
            },

            .while_loop => |loop| {
                const loop_start = try cg.create_label(.here);
                const loop_end = try cg.create_label(.undefined);
                loop_start.used = true;
                loop_end.used = true;

                try cg.emit(.{
                    .false_branch = .{
                        .target = loop_end.id,
                        .condition = loop.condition,
                    },
                });
                try cg.generate_block(loop.body, .{
                    .scope = ctx.scope,
                    .is_toplevel = ctx.is_toplevel,
                    .loop = .{
                        .cont_point = loop_start,
                        .break_point = loop_end,
                    },
                });
                try cg.emit(.{
                    .branch = .{ .target = loop_start.id },
                });

                try cg.define_label(loop_end);
            },
            .if_condition => |cond| {
                const cond_skip = try cg.create_label(.undefined);
                cond_skip.used = true;

                try cg.emit(.{
                    .false_branch = .{
                        .target = cond_skip.id,
                        .condition = cond.condition,
                    },
                });

                try cg.generate_block(cond.true_body, ctx);

                try cg.define_label(cond_skip);
            },

            .@"break" => {
                const loop = ctx.loop orelse return error.BreakOutsideLoop;
                loop.break_point.used = true;
                try cg.emit(.{
                    .branch = .{ .target = loop.break_point.id },
                });
            },
            .@"continue" => {
                const loop = ctx.loop orelse return error.BreakOutsideLoop;
                loop.cont_point.used = true;
                try cg.emit(.{
                    .branch = .{ .target = loop.cont_point.id },
                });
            },
        }
    }

    fn get_return_value(cg: *CodeGen, scope: []const u8) ![]const u8 {
        return try std.fmt.allocPrint(cg.allocator, "{s}:retval", .{
            scope,
        });
    }

    fn emit(cg: *CodeGen, instr: ir.Instruction) !void {
        try cg.instructions.append(instr);
    }
};

fn render(allocator: std.mem.Allocator, pgm: ir.Program, sm: ir.StateMachine, stream: anytype) !void {
    const fmt_id = std.zig.fmtId;

    const Renderer = struct {
        const Renderer = @This();

        const Writer = @TypeOf(stream);

        const State = enum(u64) {
            initial = 0,

            stopped = std.math.maxInt(u64),
            _,

            pub fn format(state: State, fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
                _ = fmt;
                _ = options;
                switch (state) {
                    .initial => try writer.writeAll("initial"),
                    .stopped => try writer.writeAll("stopped"),
                    _ => try writer.print("state{}", .{@intFromEnum(state)}),
                }
            }
        };

        writer: Writer,
        states: usize = 0,
        program: ir.Program,
        sm: ir.StateMachine,

        offset_to_state: std.AutoArrayHashMap(usize, State),
        label_to_state: std.AutoArrayHashMap(ir.Label, State),

        current_state: ?State = null,
        needs_stop_state: bool = false,

        required_states: std.StringArrayHashMap(void),
        required_asyncs: std.StringArrayHashMap(void),

        fn alloc_state(ren: *Renderer) !State {
            ren.states += 1;
            return @enumFromInt(ren.states);
        }

        fn state_from_offset(ren: *Renderer, offset: usize) ?State {
            return ren.offset_to_state.get(offset);
        }

        fn state_from_label(ren: *Renderer, lbl: ir.Label) State {
            return ren.label_to_state.get(lbl).?;
        }

        fn run(ren: *Renderer) !void {
            try ren.writeAll("\n\n");
            try ren.writeAll("pub const SM = struct {\n");

            try ren.writeAll("  state: State = .initial,\n");
            try ren.writeAll("  branches: BranchSet = .{},\n");
            try ren.writeAll("  data: Data = .{},\n");

            try ren.writeAll("  pub fn step(sm: *SM, resume_val: ReturnValue) !Result {\n");

            try ren.writeAll("    __sm__: switch(sm.state) {\n");

            try ren.offset_to_state.putNoClobber(0, .initial);

            for (ren.sm.labels.values()) |value| {
                const offset = value.offset.?;

                const gop = try ren.offset_to_state.getOrPut(offset);
                if (gop.found_existing)
                    continue;
                gop.value_ptr.* = try ren.alloc_state();
            }

            for (ren.sm.labels.keys(), ren.sm.labels.values()) |key, value| {
                try ren.label_to_state.putNoClobber(key, ren.offset_to_state.get(value.offset.?).?);
            }

            var last_result: RenderResult = .default;
            for (ren.sm.instructions, 0..) |instr, offset| {
                if (ren.offset_to_state.get(offset)) |state| {
                    try ren.write_new_state(state, (last_result != .switches_state));
                }

                last_result = try ren.render_instr(instr);
            }

            if (ren.needs_stop_state) {
                try ren.write_new_state(.stopped, (last_result != .switches_state));
                try ren.writeAll("        @panic(\"The state machine has stopped and can no longer be resumed!\");\n");
            }

            if (ren.current_state != null) {
                try ren.writeAll("      },\n");
            }
            try ren.writeAll("    }\n");

            try ren.writeAll("  }\n\n");

            try ren.writeAll("  const State = enum {\n");
            for (0..ren.states + 1) |state_id| {
                const state: State = @enumFromInt(state_id);
                try ren.print("    {},\n", .{state});
            }
            if (ren.needs_stop_state) {
                try ren.writeAll("    stopped,\n");
            }
            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const BranchSet = struct {};\n\n");
            try ren.writeAll("  const Data = struct {\n");

            for (ren.required_states.keys()) |state_name| {
                try ren.print("{}: u8 = undefined,\n", .{
                    fmt_id(state_name),
                });
            }

            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const ReturnValue = union(enum) {\n");
            try ren.writeAll("      launch,\n\n");
            for (ren.required_asyncs.keys()) |async_func| {
                const suspender = ren.program.suspenders.get(async_func).?;

                try ren.print("      {}: {},\n", .{
                    fmt_id(async_func),
                    fmt_raw(suspender.return_type),
                });
            }
            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const Result = union(enum) {\n");
            try ren.writeAll("      stop,\n\n");
            for (ren.required_asyncs.keys()) |async_func| {
                const suspender = ren.program.suspenders.get(async_func).?;

                try ren.print("      {}: struct{{", .{
                    fmt_id(async_func),
                });
                for (suspender.parameters, 0..) |param, index| {
                    if (index > 0)
                        try ren.writeAll(", ");
                    try ren.print("{}", .{fmt_raw(param.type)});
                }
                try ren.writeAll("},\n");
            }

            try ren.writeAll("  };\n\n");

            try ren.writeAll("\n};\n\n");
        }

        const RenderResult = enum {
            default,
            switches_state,
        };
        fn render_instr(ren: *Renderer, instr: ir.Instruction) !RenderResult {
            try ren.print("      // <{s}>\n", .{@tagName(instr)});
            const output = try ren.render_instr_inner(instr);
            try ren.print("      // </{s}>\n\n", .{@tagName(instr)});
            return output;
        }

        fn render_instr_inner(ren: *Renderer, instr: ir.Instruction) !RenderResult {
            switch (instr) {
                .stop => {
                    ren.needs_stop_state = true;
                    try ren.writeAll("      sm.state = .stopped;\n");
                    try ren.writeAll("      return .stop;\n");
                    return .switches_state;
                },
                .execute => |code| {
                    try ren.print("      {s}\n", .{ren.fmt_code(code)});
                    return .default;
                },
                .branch => |branch| {
                    try ren.write_state_switch(ren.state_from_label(branch.target));
                    return .switches_state;
                },
                .true_branch, .false_branch => |branch| {
                    try ren.print("      if(({s}) == {}) {{\n", .{
                        ren.fmt_code(branch.condition),
                        (instr == .true_branch),
                    });
                    try ren.write_state_switch(ren.state_from_label(branch.target));
                    try ren.writeAll("      }\n");
                    return .default;
                },

                .dyn_branch => |branch| {
                    try ren.print("// TODO: dyn_branch({s})\n", .{branch.target});
                    return .switches_state;
                },

                .call => |branch| {
                    try ren.print("// TODO: call({s})\n", .{branch.target});
                    return .switches_state;
                },

                .yield => |yield| {
                    try ren.required_asyncs.put(yield.function, {});

                    const hopstate = try ren.alloc_state();

                    try ren.print("      sm.state = .{s};\n", .{hopstate});
                    try ren.print("      return .{{ .{} = .{{ ", .{
                        fmt_id(yield.function),
                    });
                    for (yield.arguments, 0..) |arg, i| {
                        if (i > 0) {
                            try ren.writeAll(", ");
                        }
                        try ren.print("({s})", .{ren.fmt_code(arg)});
                    }
                    try ren.writeAll(" } };\n");
                    try ren.write_new_state(hopstate, false);
                    try ren.print("    if(resume_val != .{[0]})\n      @panic(\"BUG: State machine must be resumed with .{[0]}\");\n", .{
                        fmt_id(yield.function),
                    });

                    if (yield.output) |output_target| {
                        try ren.print("{s} = ", .{ren.fmt_code(output_target)});
                        if (yield.use_try) {
                            try ren.writeAll("try ");
                        }
                        try ren.print("resume_val.{};\n", .{fmt_id(yield.function)});
                    } else {
                        std.debug.assert(yield.use_try == false);
                    }

                    return .default;
                },

                .set_state => |state| {
                    try ren.print("// TODO: set_state({s}, {s})\n", .{
                        state.variable,
                        ren.fmt_code(state.value),
                    });
                    return .default;
                },
            }
        }

        fn write_state_switch(ren: *Renderer, state: State) !void {
            try ren.print("      sm.state = .{};\n", .{state});
            try ren.print("      continue :__sm__ .{};\n", .{state});
        }

        fn write_new_state(ren: *Renderer, new_state: State, include_autobranch: bool) !void {
            if (ren.current_state != null) {
                if (include_autobranch) {
                    try ren.write_state_switch(new_state);
                }
                try ren.writeAll("    },\n");
            }
            try ren.print("    .{} => {{\n", .{new_state});
            ren.current_state = new_state;
        }

        fn fmt_raw(code: ir.TextBlock) std.fmt.Formatter(format_raw) {
            return .{ .data = code };
        }

        fn format_raw(code: ir.TextBlock, fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.writeAll(code.text);
        }

        fn fmt_code(ren: *Renderer, code: ir.TextBlock) std.fmt.Formatter(format_code) {
            return .{ .data = .{ ren, code } };
        }

        fn format_code(args: struct { *Renderer, ir.TextBlock }, fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            const ren: *Renderer, const code: ir.TextBlock = args;

            _ = fmt;
            _ = options;

            const text = code.text;

            var pos: usize = 0;
            while (std.mem.indexOfPos(u8, text, pos, "${")) |index| {
                try writer.writeAll(text[pos..index]);

                const end = std.mem.indexOfScalarPos(u8, text, index + 2, '}') orelse {
                    try writer.writeAll("${");
                    pos += 2;
                    continue;
                };

                const name = text[index + 2 .. end];

                try ren.required_states.put(name, {});

                try writer.print("sm.data.{}", .{
                    fmt_id(name),
                });

                pos = end + 1;
            }

            try writer.writeAll(text[pos..]);
        }

        fn writeAll(ren: *Renderer, str: []const u8) !void {
            try ren.writer.writeAll(str);
        }

        fn print(ren: *Renderer, comptime fmt: []const u8, args: anytype) !void {
            try ren.writer.print(fmt, args);
        }
    };

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    var renderer: Renderer = .{
        .writer = stream,
        .program = pgm,
        .sm = sm,
        .offset_to_state = .init(arena.allocator()),
        .label_to_state = .init(arena.allocator()),
        .required_states = .init(arena.allocator()),
        .required_asyncs = .init(arena.allocator()),
    };

    try renderer.run();
}
