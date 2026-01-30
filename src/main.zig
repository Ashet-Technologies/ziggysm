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
    const stdout = std.fs.File.stdout().deprecatedWriter();

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

    try ast.dump(document, stdout);

    var program = try generate_code(allocator, document);
    defer program.deinit();
    {
        const writer = stdout;
        for (program.structure) |node| {
            switch (node) {
                .state_machine => |sm| {
                    try writer.print("StateMachine {s}:\n", .{sm.name});
                    try ir.dump(sm, writer);
                },
                .top_level_code => |code| try writer.print("TopLevel Code: {f}\n", .{code}),
            }
        }
    }

    const unformatted_code = blk: {
        var sink: std.ArrayList(u8) = .empty;
        defer sink.deinit(allocator);

        for (program.structure) |node| {
            switch (node) {
                .state_machine => |sm| try render(allocator, program, sm, sink.writer(allocator)),
                .top_level_code => |code| try sink.appendSlice(allocator, code.text), // no processing of global replacements
            }
        }

        break :blk try sink.toOwnedSliceSentinel(allocator, 0);
    };

    const formatted_code = blk: {
        const zig_ast = try std.zig.Ast.parse(allocator, unformatted_code, .zig);
        if (zig_ast.errors.len > 0)
            break :blk unformatted_code;

        break :blk try zig_ast.renderAlloc(allocator);
    };

    if (cli.options.output.len > 0) {
        try std.fs.cwd().writeFile(.{
            .sub_path = cli.options.output,
            .data = formatted_code,
        });
    } else {
        try stdout.writeAll(formatted_code);
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
    suspender,
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

        std.log.err("{}:{}: error.{s} at {f}", .{
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

    var list: std.ArrayList(ast.TopLevelNode) = .empty;
    defer list.deinit(parser.allocator);

    while (try parser.accept_top_level_node()) |tln| {
        if (tln == .raw_code and tln.raw_code.text.len == 0)
            continue;
        try list.append(parser.allocator, tln);
    }

    const top_level_nodes = try list.toOwnedSlice(parser.allocator);
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

            .suspender => {
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
                    .suspender = .{
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
                return .{ .procedure = sm };
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
        var statements: std.ArrayList(ast.Statement) = .empty;
        defer statements.deinit(parser.allocator);

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
            try statements.append(parser.allocator, stmt);
        }

        std.debug.assert(tok.data[tok.pos - 1] == '}');

        return .{
            .statements = try statements.toOwnedSlice(parser.allocator),
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
                if (std.mem.eql(u8, keyword, "$loop")) {
                    const body = try parser.accept_block();

                    return .{ .infinite_loop = .{
                        .body = body,
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

        var list: std.ArrayList(ast.Parameter) = .empty;
        defer list.deinit(parser.allocator);

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

            try list.append(parser.allocator, .{
                .name = name,
                .type = .{ .text = type_name },
            });

            if (!parser.try_accept_literal(","))
                break;
        }

        try parser.accept_literal(")");

        return try list.toOwnedSlice(parser.allocator);
    }

    pub fn next_parens_list(parser: *Parser) ![]ast.TextBlock {
        const tokenizer: *Tokenizer = parser.tokenizer;

        try parser.accept_literal("(");

        var list: std.ArrayList(ast.TextBlock) = .empty;
        defer list.deinit(parser.allocator);

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
                try list.append(parser.allocator, .{ .text = group });
            }

            if (!parser.try_accept_literal(","))
                break;
        }

        try parser.accept_literal(")");

        return try list.toOwnedSlice(parser.allocator);
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

            if (std.mem.eql(u8, keyword, "$suspender"))
                return .{ .type = .suspender, .text = keyword };

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
        procedure: StateMachine,
        suspender: Suspender,
    };

    pub const Parameter = struct {
        name: []const u8,
        type: TextBlock,
    };

    pub const Suspender = struct {
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
        infinite_loop: InfiniteLoop,
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

    pub const InfiniteLoop = struct {
        body: Block,
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

        pub fn format(tb: TextBlock, writer: *std.io.Writer) std.io.Writer.Error!void {
            const max_len = 32;
            const split_len = @divExact(max_len, 2);
            try writer.writeAll("TextBlock('");
            if (tb.text.len < max_len) {
                try std.zig.stringEscape(tb.text, writer);
            } else {
                const head = tb.text[0..split_len];
                const tail = tb.text[tb.text.len - split_len ..];
                try std.zig.stringEscape(head, writer);
                try writer.writeAll("…");
                try std.zig.stringEscape(tail, writer);
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
                        .raw_code => |code| try dmp.stream.print("{f}\n", .{code}),

                        .suspender => |func| {
                            try dmp.stream.print("suspender {s}({any}) {f}\n", .{
                                func.name,
                                func.parameters,
                                func.return_type,
                            });
                        },

                        .procedure, .state_machine, .sub_machine => |sm| {
                            try dmp.stream.print("{s} {s}({any}) {f}\n", .{ @tagName(node), sm.name, sm.parameters, sm.return_type });
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
                    .raw_code => |code| try dmp.stream.print("{f}\n", .{code}),
                    .state_variable => |state| {
                        try dmp.stream.print("$state {s}: {f} = {f};\n", .{
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
                            try dmp.stream.print("-> {s}{f}", .{
                                if (call.with_try) " $state" else "",
                                output,
                            });
                        }
                        try dmp.stream.writeAll(";\n");
                    },
                    .@"return" => |ret| {
                        try dmp.stream.writeAll("$return");
                        if (ret.value) |value| {
                            try dmp.stream.print(" {f}", .{value});
                        }

                        try dmp.stream.writeAll(";\n");
                    },
                    .infinite_loop => |cond| {
                        try dmp.stream.writeAll("$loop\n");
                        try dmp.block(cond.body, depth);
                    },
                    .while_loop => |cond| {
                        try dmp.stream.writeAll("$while_loop(");
                        try dmp.stream.print("{f}", .{cond.condition});
                        try dmp.stream.writeAll(")\n");
                        try dmp.block(cond.body, depth);
                    },
                    .if_condition => |cond| {
                        try dmp.stream.writeAll("$if_condition(");
                        try dmp.stream.print("{f}", .{cond.condition});
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
    pub const TextBlock = struct {
        text: []const u8,

        /// required to correctly tag state variables inside the block
        /// `null` is in the global scope
        scope: ?[]const u8,

        pub fn global(tb: ast.TextBlock) TextBlock {
            return .{ .text = tb.text, .scope = null };
        }

        pub fn local(tb: ast.TextBlock, scope: []const u8) TextBlock {
            return .{ .text = tb.text, .scope = scope };
        }

        pub fn format(tb: TextBlock, writer: *std.io.Writer) std.io.Writer.Error!void {
            const max_len = 32;
            const split_len = @divExact(max_len, 2);
            try writer.writeAll("TextBlock(");
            if (tb.scope) |scope| {
                try writer.writeAll("'");
                try std.zig.stringEscape(scope, writer);
                try writer.writeAll("', ");
            } else {
                try writer.writeAll("null, ");
            }
            if (tb.text.len < max_len) {
                try writer.writeAll("'");
                try std.zig.stringEscape(tb.text, writer);
                try writer.writeAll("'");
            } else {
                const head = tb.text[0..split_len];
                const tail = tb.text[tb.text.len - split_len ..];
                try writer.writeAll("'");
                try std.zig.stringEscape(head, writer);
                try writer.writeAll("…");
                try std.zig.stringEscape(tail, writer);
                try writer.writeAll("'");
            }
            try writer.writeAll("')");
        }
    };

    pub const Label = enum(usize) {
        undefined = std.math.maxInt(usize),
        _,

        pub fn format(lbl: Label, writer: *std.io.Writer) std.io.Writer.Error!void {
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
    };

    pub const Parameter = struct {
        name: []const u8,
        type: TextBlock,
    };

    pub const StateMachine = struct {
        name: []const u8,

        labels: std.AutoArrayHashMap(Label, LabelData), // id => label info
        submachines: std.StringArrayHashMap(Label), // string => id
        states: std.StringArrayHashMap(TextBlock), // name => type
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
        pub const Tag = @typeInfo(Instruction).@"union".tag_type.?;

        stop,
        execute: TextBlock,

        branch: Branch,
        true_branch: ConditionalBranch,
        false_branch: ConditionalBranch,

        call: Call,
        ret: Return,

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

    pub const Call = struct {
        target: Label,
        dest_variable: []const u8,
    };

    pub const Return = struct {
        dest_variable: []const u8,
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
                try writer.print("        {f}:\n", .{lbl});
            }
            try writer.print("0x{X:0>4}    ", .{
                offset,
            });
            switch (instr) {
                .stop => try writer.writeAll("STOP"),
                .execute => |code| {
                    try writer.print("EXEC  {f}", .{code});
                },
                .yield => |yield| {
                    try writer.print("YIELD {s}, {any}, {s}{?f}", .{
                        yield.function,
                        yield.arguments,
                        if (yield.use_try) "try " else "",
                        yield.output,
                    });
                },
                .set_state => |state| {
                    try writer.print("SETST {s}, {f}", .{ state.variable, state.value });
                },
                .branch => |branch| {
                    try writer.print("BR    {f}", .{branch.target});
                },
                .call => |branch| {
                    try writer.print("CALL  {f}, {s}", .{ branch.target, branch.dest_variable });
                },
                .ret => |branch| {
                    try writer.print("RET   {s}", .{branch.dest_variable});
                },
                .true_branch => |branch| {
                    try writer.print("BR.T  {f}, {f}", .{ branch.target, branch.condition });
                },
                .false_branch => |branch| {
                    try writer.print("BR.F  {f}, {f}", .{ branch.target, branch.condition });
                },
            }
            try writer.writeAll("\n");
        }
    }
};

fn generate_code(allocator: std.mem.Allocator, document: ast.Document) !ir.Program {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();

    var structure: std.ArrayList(ir.Node) = .empty;

    var suspenders: std.StringArrayHashMap(ir.Suspender) = .init(arena.allocator());
    var submachines: std.StringArrayHashMap(CodeGen.StateMachine) = .init(arena.allocator());

    // Create an index for all process, suspenders and submachines:
    for (document.top_level_nodes) |node| {
        switch (node) {
            .raw_code => continue,

            .suspender => |suspender| {
                const previous = try suspenders.fetchPut(suspender.name, .{
                    .name = suspender.name,
                    .parameters = try CodeGen.transpose_param_list(arena.allocator(), suspender.parameters),
                    .return_type = .global(suspender.return_type),
                });
                if (previous != null)
                    return error.DuplicateSuspender;
            },
            .state_machine, .sub_machine, .procedure => |sm| {
                const previous = try submachines.fetchPut(sm.name, .{
                    .name = sm.name,
                    .parameters = try CodeGen.transpose_param_list(arena.allocator(), sm.parameters),
                    .return_type = .global(sm.return_type),
                    .sm = sm,
                    .kind = switch (node) {
                        .state_machine => .statemachine,
                        .sub_machine => .submachine,
                        .procedure => .procedure,
                        .raw_code, .suspender => unreachable,
                    },
                });
                if (previous != null)
                    return error.DuplicateStateMachine;
            },
        }
    }

    for (document.top_level_nodes) |node| {
        const sm = switch (node) {
            .state_machine => |*sm| sm,
            .raw_code => |code| {
                try structure.append(arena.allocator(), .{ .top_level_code = .global(code) });
                continue;
            },

            .suspender, .procedure, .sub_machine => continue,
        };

        var cg: CodeGen = .{
            .allocator = arena.allocator(),
            .instructions = .empty,
            .labels = .empty,
            .tagged_labels = .init(arena.allocator()),
            .suspenders = &suspenders,
            .submachines = &submachines,
            .emitted_machines = .init(arena.allocator()),
            .required_machines = .init(arena.allocator()),
            .state_variables = .init(arena.allocator()),
        };

        _ = try cg.generate(sm.*, .statemachine);

        // Generate all dependencies as well here:
        while (cg.required_machines.pop()) |kv| {
            const submachine = submachines.get(kv.key) orelse return error.MissingDependency;
            _ = try cg.generate(submachine.sm, submachine.kind);
        }
        std.debug.assert(cg.required_machines.count() == 0);
        std.debug.assert(cg.emitted_machines.count() > 0);

        const instructions = try cg.instructions.toOwnedSlice(cg.allocator);
        errdefer cg.allocator.free(instructions);

        var labels: std.AutoArrayHashMap(ir.Label, ir.LabelData) = .init(arena.allocator());
        errdefer labels.deinit();

        for (cg.labels.items) |label| {
            try labels.put(label.id, .{
                .offset = label.offset,
                .tag = null,
            });
        }

        const generated_sm: ir.StateMachine = .{
            .name = sm.name,
            .instructions = instructions,
            .labels = labels,
            .submachines = cg.tagged_labels,
            .states = cg.state_variables,
        };

        try structure.append(arena.allocator(), .{ .state_machine = generated_sm });
    }

    return .{
        .arena = arena,
        .structure = try structure.toOwnedSlice(arena.allocator()),
        .suspenders = suspenders,
    };
}

const CodeGen = struct {
    pub const StateMachine = struct {
        pub const Kind = enum {
            statemachine,
            submachine,
            procedure,

            fn is_top_level(kind: Kind) bool {
                return switch (kind) {
                    .procedure => false,
                    .statemachine => true,
                    .submachine => true,
                };
            }
        };

        name: []const u8,
        parameters: []const ir.Parameter,
        return_type: ir.TextBlock,
        sm: ast.StateMachine,
        kind: Kind,
    };

    const Label = struct {
        id: ir.Label,
        offset: ?usize = null,
        used: bool = false,
        has_tag: bool = false,
    };

    allocator: std.mem.Allocator,
    suspenders: *std.StringArrayHashMap(ir.Suspender),
    submachines: *std.StringArrayHashMap(StateMachine),

    emitted_machines: std.StringArrayHashMap(void),
    required_machines: std.StringArrayHashMap(void),

    state_variables: std.StringArrayHashMap(ir.TextBlock), // name => type

    instructions: std.ArrayList(ir.Instruction),

    labels: std.ArrayList(*Label),
    tagged_labels: std.StringArrayHashMap(ir.Label),

    fn transpose_block_list(cg: *CodeGen, blocks: []const ast.TextBlock, scope: []const u8) ![]ir.TextBlock {
        const array = try cg.allocator.alloc(ir.TextBlock, blocks.len);
        for (blocks, array) |src, *dst| {
            dst.* = .local(src, scope);
        }
        return array;
    }

    fn transpose_param_list(allocator: std.mem.Allocator, blocks: []const ast.Parameter) ![]ir.Parameter {
        const array = try allocator.alloc(ir.Parameter, blocks.len);
        for (blocks, array) |src, *dst| {
            dst.* = .{
                .name = src.name,
                .type = .global(src.type),
            };
        }
        return array;
    }

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
        try cg.labels.append(cg.allocator, lbl);

        return lbl;
    }

    /// Gets or creates a new label with a given tag.
    fn get_tagged_label(cg: *CodeGen, tag: []const u8) !*Label {
        const gop = try cg.tagged_labels.getOrPut(tag);
        if (gop.found_existing) {
            return cg.labels.items[@intFromEnum(gop.value_ptr.*)];
        }
        errdefer _ = cg.tagged_labels.swapRemove(tag);
        const label = try cg.create_label(.undefined);
        gop.value_ptr.* = label.id;
        return label;
    }

    fn define_label(cg: *CodeGen, lbl: *Label) !void {
        if (lbl.offset != null)
            @panic("label was defined twice!");
        lbl.offset = cg.instructions.items.len;
    }

    fn add_sm_dependency(cg: *CodeGen, name: []const u8) !void {
        if (cg.emitted_machines.contains(name))
            return;
        try cg.required_machines.put(name, {});
    }

    fn generate(cg: *CodeGen, sm: ast.StateMachine, sm_kind: CodeGen.StateMachine.Kind) !ir.Label {
        try cg.emitted_machines.put(sm.name, {});
        _ = cg.required_machines.swapRemove(sm.name);

        const sm_label = try cg.get_tagged_label(sm.name);
        try cg.define_label(sm_label);

        const ctx: BlockContext = .{
            .scope = sm.name,
            .sm_kind = sm_kind,
        };

        if (sm_kind == .procedure) {
            try cg.add_state_var(try cg.get_state_name(sm.name, .return_value), .global(sm.return_type));
        }

        for (sm.parameters) |param| {
            try cg.add_state_var(try cg.get_state_name(sm.name, .{ .parameter = param.name }), .global(param.type));
        }

        try cg.generate_block(sm.body, ctx);

        const last_instr = if (cg.instructions.items.len > 0)
            @as(ir.Instruction.Tag, cg.instructions.items[cg.instructions.items.len - 1])
        else
            null;

        const last_is_branch = (last_instr == .stop or last_instr == .branch or last_instr == .ret);

        const any_jumps_to_end = for (cg.labels.items) |lbl| {
            if (lbl.offset == cg.instructions.items.len)
                break true;
        } else false;

        // We do need an additional STOP/RETURN instruction if
        // the previous instruction was not a branch (and would fall through otherwise)
        // and we also need it if any part of the code jumps to the end of the code
        // (which would also "fall through" then):
        if (!last_is_branch or any_jumps_to_end) {
            // last statement is an implicit return, which turns into STOP
            // for statemachine and submachine and into RET for procs
            try cg.generate_stmt(.{ .@"return" = .{ .value = null } }, ctx);
        }

        // Assert all labels point to at least one instruction:
        for (cg.labels.items) |lbl| {
            if (lbl.offset) |offset| {
                std.debug.assert(offset < cg.instructions.items.len);
            }
        }

        return sm_label.id;
    }

    const BlockContext = struct {
        const Loop = struct {
            break_point: *Label,
            cont_point: *Label,
        };

        scope: []const u8,
        loop: ?Loop = null,
        sm_kind: StateMachine.Kind,

        pub fn local(ctx: BlockContext, block: ast.TextBlock) ir.TextBlock {
            return .local(block, ctx.scope);
        }
    };

    fn generate_block(cg: *CodeGen, block: ast.Block, ctx: BlockContext) error{
        OutOfMemory,
        SyntaxError,
        BreakOutsideLoop,
        ValueReturnOnToplevel,
        UnknownSuspender,
        ParameterMismatch,
        UnknownStateMachine,
        DuplicateState,
    }!void {
        for (block.statements) |stmt| {
            try cg.generate_stmt(stmt, ctx);
        }
    }

    fn add_state_var(cg: *CodeGen, name: []const u8, type_name: ir.TextBlock) !void {
        std.log.info("var {s}: {f}", .{ name, type_name });
        const gop = try cg.state_variables.getOrPut(name);
        if (gop.found_existing) {
            try cg.emit_error("duplicate state {s}", .{name});
            return error.DuplicateState;
        }
        gop.value_ptr.* = type_name;
    }

    fn error_union_payload(type_name: ir.TextBlock) ?ir.TextBlock {
        const text = type_name.text;
        var paren_depth: usize = 0;
        var brace_depth: usize = 0;
        var bracket_depth: usize = 0;

        for (text, 0..) |ch, idx| {
            switch (ch) {
                '(' => paren_depth += 1,
                ')' => {
                    if (paren_depth > 0) {
                        paren_depth -= 1;
                    }
                },
                '{' => brace_depth += 1,
                '}' => {
                    if (brace_depth > 0) {
                        brace_depth -= 1;
                    }
                },
                '[' => bracket_depth += 1,
                ']' => {
                    if (bracket_depth > 0) {
                        bracket_depth -= 1;
                    }
                },
                '!' => {
                    if (paren_depth == 0 and brace_depth == 0 and bracket_depth == 0) {
                        const payload = std.mem.trim(u8, text[idx + 1 ..], " \t\n\r");
                        if (payload.len == 0) {
                            return null;
                        }
                        return .{ .text = payload, .scope = type_name.scope };
                    }
                },
                else => {},
            }
        }

        return null;
    }

    fn emit_error(cg: *CodeGen, comptime fmt: []const u8, args: anytype) !void {
        std.log.err(fmt, args);
        _ = cg;
    }

    fn generate_stmt(cg: *CodeGen, stmt: ast.Statement, ctx: BlockContext) !void {
        switch (stmt) {
            .raw_code => |code| try cg.emit(.{ .execute = ctx.local(code) }),

            .state_variable => |state| {
                const state_name = try cg.get_state_name(ctx.scope, .{ .local = state.variable_name });

                try cg.add_state_var(state_name, .global(state.type_spec));
                try cg.emit(.{ .set_state = .{
                    .variable = state_name,
                    .value = ctx.local(state.initial_value),
                } });
            },

            .yield => |yield| {
                const suspender = cg.suspenders.get(yield.function_name) orelse {
                    try cg.emit_error("Unknown suspender {s}", .{yield.function_name});
                    return error.UnknownSuspender;
                };

                if (yield.arguments.len != suspender.parameters.len) {
                    try cg.emit_error("parameter mismatch for suspender {s}: expecteed {} parameters, found {}", .{
                        suspender.name,
                        suspender.parameters.len,
                        yield.arguments.len,
                    });
                    return error.ParameterMismatch;
                }

                if (yield.output_to) |output_to| {
                    if (yield.as_state) {
                        const state_type = if (yield.with_try)
                            error_union_payload(suspender.return_type) orelse suspender.return_type
                        else
                            suspender.return_type;
                        try cg.add_state_var(
                            try cg.get_state_name(ctx.scope, .{ .local = output_to.text }),
                            state_type,
                        );
                    }
                }

                try cg.emit(.{ .yield = .{
                    .function = yield.function_name,
                    .arguments = try cg.transpose_block_list(yield.arguments, ctx.scope),
                    .output = if (yield.output_to) |output_to|
                        if (yield.as_state)
                            ctx.local(.{ .text = try std.fmt.allocPrint(cg.allocator, "${{{s}}}", .{output_to.text}) })
                        else
                            ctx.local(output_to)
                    else
                        null,
                    .use_try = yield.with_try,
                } });
            },

            .call => |call| {
                const submachine = cg.submachines.get(call.function_name) orelse {
                    try cg.emit_error("Unknown procedure {s}", .{call.function_name});
                    return error.UnknownStateMachine;
                };
                if (submachine.parameters.len != call.arguments.len) {
                    try cg.emit_error("parameter mismatch for procedure {s}: expecteed {} parameters, found {}", .{
                        submachine.name,
                        submachine.parameters.len,
                        call.arguments.len,
                    });
                    return error.ParameterMismatch;
                }

                for (submachine.parameters, call.arguments) |param, arg| {
                    try cg.emit(.{ .set_state = .{
                        .variable = try cg.get_state_name(submachine.name, .{ .parameter = param.name }),
                        .value = ctx.local(arg),
                    } });
                }

                if (call.output_to) |output_to| {
                    if (call.as_state) {
                        try cg.add_state_var(
                            try cg.get_state_name(ctx.scope, .{ .local = output_to.text }),
                            submachine.return_type,
                        );
                    }
                }

                const target = try cg.get_tagged_label(call.function_name);
                target.used = true;

                try cg.emit(.{ .call = .{
                    .target = target.id,
                    .dest_variable = call.function_name,
                } });

                try cg.add_sm_dependency(call.function_name);
            },

            .jump => |jmp| {
                if (jmp.with_try or jmp.as_state)
                    return error.SyntaxError;

                const submachine = cg.submachines.get(jmp.function_name) orelse {
                    try cg.emit_error("Unknown statemachine {s}", .{jmp.function_name});
                    return error.UnknownStateMachine;
                };
                if (submachine.parameters.len != jmp.arguments.len) {
                    try cg.emit_error("parameter mismatch for submachine {s}: expecteed {} parameters, found {}", .{
                        submachine.name,
                        submachine.parameters.len,
                        jmp.arguments.len,
                    });
                    return error.ParameterMismatch;
                }

                for (submachine.parameters, jmp.arguments) |param, arg| {
                    try cg.emit(.{ .set_state = .{
                        .variable = try cg.get_state_name(submachine.name, .{ .parameter = param.name }),
                        .value = ctx.local(arg),
                    } });
                }

                const target = try cg.get_tagged_label(jmp.function_name);
                target.used = true;
                try cg.emit(.{
                    .branch = .{ .target = target.id },
                });

                try cg.add_sm_dependency(jmp.function_name);
            },

            .@"return" => |ret| {
                if (ctx.sm_kind.is_top_level()) {
                    if (ret.value != null) {
                        try cg.emit_error("Cannot return a value from statemachine or submachine", .{});
                        return error.ValueReturnOnToplevel;
                    }
                    try cg.emit(.stop);
                } else {
                    if (ret.value) |value| {
                        try cg.emit(.{ .set_state = .{
                            .variable = try cg.get_state_name(ctx.scope, .return_value),
                            .value = ctx.local(value),
                        } });
                    }
                    try cg.emit(.{ .ret = .{
                        .dest_variable = ctx.scope,
                    } });
                }
            },

            .infinite_loop => |loop| {
                const loop_start = try cg.create_label(.here);
                const loop_end = try cg.create_label(.undefined);
                loop_start.used = true;

                try cg.generate_block(loop.body, .{
                    .scope = ctx.scope,
                    .sm_kind = ctx.sm_kind,
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

            .while_loop => |loop| {
                const loop_start = try cg.create_label(.here);
                const loop_end = try cg.create_label(.undefined);
                loop_start.used = true;
                loop_end.used = true;

                try cg.emit(.{
                    .false_branch = .{
                        .target = loop_end.id,
                        .condition = ctx.local(loop.condition),
                    },
                });
                try cg.generate_block(loop.body, .{
                    .scope = ctx.scope,
                    .sm_kind = ctx.sm_kind,
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
                        .condition = ctx.local(cond.condition),
                    },
                });

                try cg.generate_block(cond.true_body, ctx);

                try cg.define_label(cond_skip);
            },

            .@"break" => {
                const loop = ctx.loop orelse {
                    try cg.emit_error("Cannot use $break outside of a loop.", .{});
                    return error.BreakOutsideLoop;
                };
                loop.break_point.used = true;
                try cg.emit(.{
                    .branch = .{ .target = loop.break_point.id },
                });
            },
            .@"continue" => {
                const loop = ctx.loop orelse {
                    try cg.emit_error("Cannot use $continue outside of a loop.", .{});
                    return error.BreakOutsideLoop;
                };
                loop.cont_point.used = true;
                try cg.emit(.{
                    .branch = .{ .target = loop.cont_point.id },
                });
            },
        }
    }

    fn get_state_name(
        cg: *CodeGen,
        scope: []const u8,
        state: union(enum) {
            return_value,
            parameter: []const u8,
            local: []const u8,
        },
    ) ![]const u8 {
        return switch (state) {
            .return_value => scope,
            .parameter, .local => |tag| try std.fmt.allocPrint(cg.allocator, "{s}:{s}", .{
                scope,
                tag,
            }),
        };
    }

    fn emit(cg: *CodeGen, instr: ir.Instruction) !void {
        try cg.instructions.append(cg.allocator, instr);
    }
};

fn render(allocator: std.mem.Allocator, pgm: ir.Program, sm: ir.StateMachine, stream: anytype) !void {
    const fmt_id = std.zig.fmtId;

    const Renderer = struct {
        const Renderer = @This();

        const Writer = @TypeOf(stream);

        const StateType = enum(u4) {
            builtin = 0,
            offset = 1,
            yield = 2,
            branch = 3,
        };

        const State = enum(u64) {
            const Mask = packed struct(u64) { index: u60, type: StateType };

            initial = 0,
            stopped = 1,

            _,

            fn get_type(s: State) StateType {
                const mask: Mask = @bitCast(s);
                return mask.type;
            }

            pub fn format(state: State, writer: *std.io.Writer) std.io.Writer.Error!void {
                switch (state) {
                    .initial => try writer.writeAll("initial"),
                    .stopped => try writer.writeAll("stopped"),
                    _ => {
                        const mask: Mask = @bitCast(@intFromEnum(state));
                        try writer.print("{s}{}", .{ @tagName(mask.type), mask.index });
                    },
                }
            }
        };

        writer: Writer,
        allocator: std.mem.Allocator,
        states: std.AutoArrayHashMap(State, void),

        program: ir.Program,
        sm: ir.StateMachine,

        offset_to_state: std.AutoArrayHashMap(usize, State),
        label_to_state: std.AutoArrayHashMap(ir.Label, State),

        current_state: ?State = null,
        needs_stop_state: bool = false,

        required_states: std.StringArrayHashMap(void),
        required_suspenders: std.StringArrayHashMap(void),

        required_dynbranch: std.StringArrayHashMap(void),

        fn alloc_state(ren: *Renderer, state_type: StateType) !State {
            std.debug.assert(state_type != .builtin);
            const mask: State.Mask = .{
                .type = state_type,
                .index = @intCast(ren.states.count()),
            };
            const state: State = @enumFromInt(@as(u64, @bitCast(mask)));
            try ren.states.put(state, {});
            return state;
        }

        fn alloc_state_from_offset(ren: *Renderer, offset: usize) !State {
            const mask: State.Mask = .{
                .type = .offset,
                .index = @intCast(offset),
            };
            const state: State = @enumFromInt(@as(u64, @bitCast(mask)));
            try ren.states.put(state, {});
            return state;
        }

        fn state_from_offset(ren: *Renderer, offset: usize) ?State {
            return ren.offset_to_state.get(offset);
        }

        fn state_from_label(ren: *Renderer, lbl: ir.Label) State {
            return ren.label_to_state.get(lbl).?;
        }

        fn run(ren: *Renderer) !void {
            try ren.writeAll("\n\n");
            try ren.print("pub const {f} = struct {{\n", .{fmt_id(ren.sm.name)});

            try ren.writeAll("  state: State = .initial,\n");
            try ren.writeAll("  branches: BranchSet = .{},\n");
            try ren.writeAll("  data: Data = .{},\n");

            try ren.print("  pub fn step(sm: *{f}, resume_val: ReturnValue) !Result {{\n", .{fmt_id(ren.sm.name)});

            try ren.writeAll("    __sm__: switch(sm.state) {\n");

            try ren.states.put(.initial, {});

            try ren.offset_to_state.putNoClobber(0, .initial);

            for (ren.sm.labels.values()) |value| {
                const offset = value.offset orelse @panic("not all labels are defined!");

                const gop = try ren.offset_to_state.getOrPut(offset);
                if (gop.found_existing)
                    continue;
                gop.value_ptr.* = try ren.alloc_state_from_offset(offset);
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
                try ren.writeAll("        if (false) break :__sm__ .stopped;\n");
                try ren.writeAll("        @panic(\"The state machine has stopped and can no longer be resumed!\");\n");
            }

            if (ren.current_state != null) {
                try ren.writeAll("      },\n");
            }
            try ren.writeAll("    }\n");

            try ren.writeAll("  }\n\n");

            try ren.writeAll("  const State = enum {\n");
            for (ren.states.keys()) |state| {
                try ren.print("    {f},\n", .{state});
            }
            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const BranchSet = struct {\n");

            for (ren.required_dynbranch.keys()) |branch| {
                try ren.print("{f}: ?State = null,\n", .{fmt_id(branch)});
            }

            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const Data = struct {\n");

            for (ren.required_states.keys()) |state_name| {
                const raw_state_type = ren.sm.states.get(state_name) orelse {
                    std.log.err("use of undeclared state {s}", .{state_name});
                    @panic("used undeclared state");
                };

                try ren.print("{f}: {f} = undefined,\n", .{
                    fmt_id(state_name),
                    fmt_raw(raw_state_type),
                });
            }

            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const ReturnValue = union(enum) {\n");
            try ren.writeAll("      launch,\n\n");
            for (ren.required_suspenders.keys()) |suspender_id| {
                const suspender = ren.program.suspenders.get(suspender_id).?;

                try ren.print("      {f}: {f},\n", .{
                    fmt_id(suspender_id),
                    fmt_raw(suspender.return_type),
                });
            }
            try ren.writeAll("  };\n\n");
            try ren.writeAll("  const Result = union(enum) {\n");
            try ren.writeAll("      stop,\n\n");
            for (ren.required_suspenders.keys()) |suspender_id| {
                const suspender = ren.program.suspenders.get(suspender_id).?;

                try ren.print("      {f}: struct{{", .{
                    fmt_id(suspender_id),
                });
                for (suspender.parameters, 0..) |param, index| {
                    if (index > 0)
                        try ren.writeAll(", ");
                    try ren.print("{f}", .{fmt_raw(param.type)});
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
                    // Adds a small "safety wrapper" around the execution blocks,
                    // and also makes the ASTgen happy
                    try ren.writeAll("    if(true) {\n");
                    try ren.writeAll("      ");
                    try ren.write_code(code);
                    try ren.writeAll("\n");
                    try ren.writeAll("    }\n");
                    return .default;
                },
                .branch => |branch| {
                    try ren.write_state_switch(ren.state_from_label(branch.target));
                    return .switches_state;
                },
                .true_branch, .false_branch => |branch| {
                    try ren.writeAll("      if((");
                    try ren.write_code(branch.condition);
                    try ren.print(") == {}) {{\n", .{(instr == .true_branch)});
                    try ren.write_state_switch(ren.state_from_label(branch.target));
                    try ren.writeAll("      }\n");
                    return .default;
                },

                .call => |call| {
                    const hopstate = try ren.alloc_state(.branch);

                    try ren.required_dynbranch.put(call.dest_variable, {});
                    try ren.print("      if(sm.branches.{f} != null) @panic(\"A recursive CALL to {f} was detected at runtime.\");\n", .{
                        fmt_id(call.dest_variable),
                        std.zig.fmtString(call.dest_variable),
                    });

                    try ren.print("      sm.branches.{f} = .{f};\n", .{ fmt_id(call.dest_variable), hopstate });

                    const target_state = ren.state_from_label(call.target);
                    try ren.write_state_switch(target_state);

                    try ren.write_new_state(hopstate, false);

                    return .default;
                },

                .ret => |ret| {
                    try ren.required_dynbranch.put(ret.dest_variable, {});

                    // 1. Fetch the return state
                    try ren.print("      sm.state = sm.branches.{f} orelse @panic(\"RET was reached without a previous CALL for proc {f}\");\n", .{
                        fmt_id(ret.dest_variable),
                        std.zig.fmtString(ret.dest_variable),
                    });
                    // 2. Reset the state such that another RET would panic
                    try ren.print("      sm.branches.{f} = null;\n", .{fmt_id(ret.dest_variable)});
                    // 3. Perform the actual branch to the return location
                    try ren.print("      continue :__sm__ sm.state;\n", .{});

                    return .switches_state;
                },

                .yield => |yield| {
                    try ren.required_suspenders.put(yield.function, {});

                    const hopstate = try ren.alloc_state(.yield);

                    try ren.print("      sm.state = .{f};\n", .{hopstate});
                    try ren.print("      return .{{ .{f} = .{{ ", .{
                        fmt_id(yield.function),
                    });
                    for (yield.arguments, 0..) |arg, i| {
                        if (i > 0) {
                            try ren.writeAll(", ");
                        }
                        try ren.writeAll("(");
                        try ren.write_code(arg);
                        try ren.writeAll(")");
                    }
                    try ren.writeAll(" } };\n");
                    try ren.write_new_state(hopstate, false);
                    try ren.print("    if(resume_val != .{[0]f})\n      @panic(\"BUG: State machine must be resumed with .{[0]f}\");\n", .{
                        fmt_id(yield.function),
                    });

                    if (yield.output) |output_target| {
                        try ren.write_code(output_target);
                    } else {
                        // Discard the value so we catch the problem when we're silenty ignoring an error:
                        try ren.writeAll("_");
                    }
                    try ren.writeAll(" = ");
                    if (yield.use_try) {
                        try ren.writeAll("try ");
                    }
                    try ren.print("resume_val.{f};\n", .{fmt_id(yield.function)});

                    return .default;
                },

                .set_state => |state| {
                    try ren.required_states.put(state.variable, {});
                    try ren.print("sm.data.{f} = (", .{fmt_id(state.variable)});
                    try ren.write_code(state.value);
                    try ren.writeAll(");\n");
                    return .default;
                },
            }
        }

        fn write_state_switch(ren: *Renderer, state: State) !void {
            try ren.print("      sm.state = .{f};\n", .{state});
            try ren.print("      continue :__sm__ .{f};\n", .{state});
        }

        fn write_new_state(ren: *Renderer, new_state: State, include_autobranch: bool) !void {
            try ren.states.put(new_state, {});
            if (ren.current_state != null) {
                if (include_autobranch) {
                    try ren.write_state_switch(new_state);
                }
                try ren.writeAll("    },\n");
            }
            try ren.print("    .{f} => {{", .{new_state});

            // render the associated labels for each state:
            {
                var any_label = false;
                for (ren.label_to_state.keys(), ren.label_to_state.values()) |key, value| {
                    if (new_state != value)
                        continue;

                    if (!any_label) {
                        try ren.writeAll(" // ");
                    } else {
                        try ren.writeAll(", ");
                    }

                    any_label = true;
                    try ren.print("{f}", .{key});
                }
            }

            try ren.writeAll("\n");
            ren.current_state = new_state;
        }

        fn fmt_raw(code: ir.TextBlock) std.fmt.Formatter(ir.TextBlock, format_raw) {
            return .{ .data = code };
        }

        fn format_raw(code: ir.TextBlock, writer: *std.io.Writer) std.io.Writer.Error!void {
            try writer.writeAll(code.text);
        }

        fn write_code(ren: *Renderer, code: ir.TextBlock) !void {
            const text = code.text;

            var pos: usize = 0;
            while (std.mem.indexOfPos(u8, text, pos, "${")) |index| {
                try ren.writer.writeAll(text[pos..index]);

                const end = std.mem.indexOfScalarPos(u8, text, index + 2, '}') orelse {
                    try ren.writer.writeAll("${");
                    pos += 2;
                    continue;
                };

                const local_name = text[index + 2 .. end];

                const global_name = try std.fmt.allocPrint(ren.allocator, "{s}:{s}", .{
                    code.scope orelse return error.NoReplacementAllowed,
                    local_name,
                });

                try ren.required_states.put(global_name, {});

                try ren.writer.print("sm.data.{f}", .{
                    fmt_id(global_name),
                });

                pos = end + 1;
            }

            try ren.writer.writeAll(text[pos..]);
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
        .allocator = arena.allocator(),
        .states = .init(arena.allocator()),
        .offset_to_state = .init(arena.allocator()),
        .label_to_state = .init(arena.allocator()),
        .required_states = .init(arena.allocator()),
        .required_suspenders = .init(arena.allocator()),
        .required_dynbranch = .init(arena.allocator()),
    };

    try renderer.run();
}
