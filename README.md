# ziggysm

ziggysm is a small compiler that transforms a `.zigsm` DSL into Zig source code
implementing state machines. You feed it a `.zigsm` file, it parses the DSL,
generates an IR, renders Zig code, formats it when possible, and writes the
result to stdout or an output file.【F:src/main.zig†L14-L82】

## Build

```bash
zig build
```

## Usage

```bash
zig build run -- path/to/example.zigsm
```

To write the generated Zig source to a file:

```bash
zig build run -- --output example.zig path/to/example.zigsm
```

## DSL syntax (overview)

The DSL describes state machines, suspenders (external calls), and control flow:

- `$statemachine Name(params) return_type { ... }`
- `$suspender name(params) return_type;`
- Statements like `$yield`, `$call`, `$jump`, `$if`, `$while`, `$loop`,
  `$return`, `$break`, `$continue`, and `$state` declarations.

See the examples in `examples/` for a fuller reference.

## Small example

```zig
const std = @import("std");

$suspender write_byte(data: u8) void;
$suspender read_byte() u8;

$statemachine Basic() void {
    $while(true) {
        $yield read_byte() -> $state output;

        $if(${output} == 0) {
            $return;
        }

        $yield write_byte(${output});
    }
}
```

This example waits for bytes, echoes them back, and stops when it reads `0`.
It matches the `examples/basic.zigsm` sample in the repo.
