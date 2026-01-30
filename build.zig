const std = @import("std");

const examples = [_][]const u8{
    "examples/basic.zigsm",
    "examples/calling.zigsm",
    "examples/call_output.zigsm",
    "examples/submachine.zigsm",
    "examples/errors.zigsm",
    "examples/loop.zigsm",
};

pub fn build(b: *std.Build) void {
    const run_step = b.step("run", "Run the app");
    const test_step = b.step("test", "Run unit tests");

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const args_dep = b.dependency("args", .{});
    const args_mod = args_dep.module("args");

    const ziggysm_mod = b.addModule("ziggysm", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "args", .module = args_mod },
        },
    });

    const exe = b.addExecutable(.{
        .name = "ziggsm",
        .root_module = ziggysm_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = ziggysm_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    test_step.dependOn(&run_exe_unit_tests.step);

    for (examples) |example_src| {
        const render_example = b.addRunArtifact(exe);
        render_example.addArg("--output");
        const example_zig = render_example.addOutputFileArg(b.fmt("{s}.zig", .{std.fs.path.basename(example_src)}));
        render_example.addFileArg(b.path(example_src));

        const example_test = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = example_zig,
                .target = target,
                .optimize = optimize,
            }),
        });

        const run_example_test = b.addRunArtifact(example_test);
        test_step.dependOn(&run_example_test.step);
    }
}
