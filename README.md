# ziggysm

ziggysm is a small Zig-flavored DSL for writing **deterministic, step-driven state machines** that cooperate with an external “driver” instead of blocking.

You write state machine logic in a direct, readable style (with explicit suspension points), then drive it by repeatedly calling a generated `.step(...)` function until the machine stops.

---

## Why

ziggysm is useful when you want:

- Protocol logic that *looks* like straight-line code, but never blocks
- Deterministic, easily testable control flow (no threads required)
- Explicit orchestration of I/O, timeouts, retries, device access, scheduling, etc.
- A single state machine instance that can be paused/resumed safely and predictably

Typical domains: embedded drivers, device protocols, game loops, cooperative schedulers, async-ish workflows without an async runtime.

---

## What it generates

A ziggysm definition generates a Zig `struct` you instantiate and drive. The generated machine has (conceptually) three parts:

- `state`: an internal enum tracking where execution will resume
- `data`: storage for `$state` variables (persistent across yields)
- `branches`: internal storage used to implement `$call`/`$return` for procedures

The driver repeatedly calls:

```zig
pub fn step(sm: *Machine, resume_val: ReturnValue) !Result
```

Each call runs the machine forward until it hits one of these boundaries:

- it yields an external request (`$yield ...`), returning a `Result.<suspender>` variant, or
- it stops (`$return;` from a top-level machine/submachine), returning `Result.stop`

Because `step()` returns `!Result`, it may also return an error if a yielded operation was marked with `$try` and the driver resumes it with an error.

---

## Core concepts

### 1) Suspenders: declaring external effects

A **suspender** describes an external action that the state machine can request. Think “an operation the outside world performs for me”.

```zig
$suspender read_byte() u8;
$suspender write_byte(data: u8) void;
```

Suspenders can return error unions too:

```zig
$suspender read_byte_with_timeout(deadline: u64) error{Timeout}!u8;
```

The state machine never calls suspenders directly; it only yields them.

---

### 2) Yielding: requesting work and pausing

`$yield` is the suspension point. When a machine yields, it returns control to the driver along with a request.

```zig
$yield read_byte() -> $state b;
$yield write_byte(${b});
```

- The first `$yield` asks the driver to perform `read_byte()` and then resumes with the returned `u8`.
- The second `$yield` asks the driver to perform `write_byte(b)` and then resumes when that action is complete.

Under the hood:
- Yielding produces a `Result.<suspender>` request containing the suspender arguments.
- Resuming must pass a `ReturnValue.<suspender>` with the suspender return value (or error union value).

If you resume with the wrong tag, the generated code panics (it treats that as a bug in the driver).

---

### 3) Persistent state: `$state` and `${...}`

Because the machine can suspend and resume, values that must live across yields must be stored in **persistent state**.

- `$state name: T = init;` declares a persistent variable stored in the machine instance.
- `-> $state name` declares a new persistent variable and assigns it the yielded result.
- `-> ${name}` assigns into an existing persistent variable.
- `${name}` reads a persistent variable inside expressions.

Example:

```zig
$state retries: u8 = 0;

$yield read_byte() -> $state b;

$if(${b} == 0) {
    ${retries} += 1;
    $return;
}
```

Important: `${name}` is a *text replacement* performed by the compiler in state-machine scopes. It expands to a field access into the machine’s `data` storage. This replacement is not performed in plain top-level code.

---

### 4) Control flow that can suspend: `$if`, `$while`, `$loop`

If a block may contain `$yield` or `$call` (anything that can suspend), use the `$...` versions of control flow.

```zig
$while(true) {
    $yield read_byte() -> $state b;
    $if(${b} == 0) { $return; }
    $yield write_byte(${b});
}
```

`$loop { ... }` is an infinite loop variant.

(You can still write normal Zig code in raw blocks, but anything that needs to suspend must use the ziggysm control forms.)

---

### 5) Procedures: `$proc` and `$call`

`$proc` defines a reusable coroutine-like routine that can yield. Use `$call` to invoke it.

```zig
$proc WriteTwice(value: u8) void {
    $yield write_byte(${value});
    $yield write_byte(${value});
}

$statemachine Example() void {
    $yield read_byte() -> $state b;
    $call WriteTwice(${b});
    $return;
}
```

#### Notes about `$call` in the current compiler
- `$call` is implemented as a CALL/RET mechanism with a stored “return state”.
- Calling the *same* procedure recursively is detected at runtime and causes a panic.
- Nested calls are fine as long as you don’t re-enter the same procedure name while it already has a pending return.

Also: `-> ...` after `$call` is currently parsed, but in the current implementation it does not move a procedure return value into that target. In other words, treat `$call` as “control flow + side effects” today.

---

### 6) Submachines and transitions: `$submachine` and `$jump`

`$submachine` defines a named mode/state that you can transition to using `$jump`.

```zig
$statemachine Main() void {
    $jump Boot();
}

$submachine Boot() void {
    // ...
    $jump Running();
}

$submachine Running() void {
    // ...
}
```

`$jump Name(args...)` performs a state transition (a branch). It does not return.

#### About `$call` with `$submachine`
The syntax allows `$call` targeting a `$submachine`, and it behaves like “jump with a remembered return address”.
However, `$return;` in a submachine stops the whole machine (it does not RET back to the caller), so in practice:
- use `$jump` for submachine transitions
- use `$proc` + `$call` for call/return-style reuse

---

## The step protocol (how to drive it)

The generated machine exposes two unions:

- `ReturnValue` (inputs to resume the machine)
- `Result` (outputs describing what the machine wants next)

### `ReturnValue`

`ReturnValue` is a tagged union containing:

- `launch` (used for the first call)
- one variant per used suspender, with the suspender’s *return type* as payload

Examples of what the driver passes back:

- For `u8` return: `. { .read_byte = 42 }`
- For `void` return: `.write_byte = {}`
- For `error{Timeout}!u8` return:
  - success: `.{ .read_byte_with_timeout = 42 }` (the error union value is “success 42”)
  - error: `.{ .read_byte_with_timeout = error.Timeout }`

### `Result`

`Result` is a tagged union containing:

- `stop`
- one variant per used suspender, where the payload is a tuple-like struct of the suspender argument types

Example yields:

- `.read_byte` (no args)
- `. { .write_byte = .{ 10 } }`

### Driving loop (typical pattern)

```zig
var sm: MyMachine = .{};
var in: MyMachine.ReturnValue = .launch;

while (true) {
    const out = try sm.step(in);

    switch (out) {
        .stop => break,

        .read_byte => {
            const b: u8 = device.readByte();
            in = .{ .read_byte = b };
        },

        .write_byte => |args| {
            device.writeByte(args[0]); // tuple-style struct
            in = .{ .write_byte = {} };
        },
    }
}
```

Notes:
- The generated code checks that `in` matches the last yielded suspender; resuming with the wrong variant is considered a bug and panics.
- `step()` may execute multiple internal instructions per call, but it only *returns* at yield/stop (or error).

---

## Error handling

If a suspender returns an error union, you choose whether the machine should propagate it automatically.

### Propagate with `$try`

`$try` compiles into `try resume_val.<suspender>` at resume time:

```zig
$yield $try read_byte_with_timeout(${deadline}) -> $state b;
```

If the driver resumes with `error.Timeout`, then `step()` returns that error immediately.

### Handle manually (advanced)

If you omit `$try`, the resume value is used without `try`. This is useful when you want custom error policy, but note:

- `$state` variables store the *success type* (not the full error union) in the current implementation.
- That means if you want to store into `$state` from an error union return, you generally must use `$try` (otherwise it won’t type-check).

For manual policies, prefer handling errors in raw Zig blocks, or design suspenders so the return type matches the policy you want to encode.

---

## Minimal examples

### A simple echo-like machine

```zig
$suspender read_byte() u8;
$suspender write_byte(data: u8) void;

$statemachine EchoUntilZero() void {
    $while(true) {
        $yield read_byte() -> $state b;
        $if(${b} == 0) { $return; }
        $yield write_byte(${b});
    }
}
```

### An endless writer

```zig
$suspender write_byte(data: u8) void;

$statemachine LoopWriteZeros() void {
    $loop {
        $yield write_byte(0);
    }
}
```

### Using `$proc` to factor logic

```zig
$suspender write_byte(data: u8) void;

$proc WriteTwice(value: u8) void {
    $yield write_byte(${value});
    $yield write_byte(${value});
}
```

---

## Testing philosophy

Because a ziggysm machine is driven by explicit `step(...)` calls, it’s straightforward to test deterministically:

1. Start with `.launch`
2. Assert which request is yielded
3. Feed back a completion value (or error)
4. Repeat until `.stop`

This makes protocol code testable without real devices, timers, threads, or I/O.

---

## Design notes

- **Determinism:** the machine only makes progress when the driver calls `step()`.
- **Explicit effects:** external work is always represented as yielded requests.
- **No hidden blocking:** the machine cannot block internally; it can only yield.
- **Driver-controlled scheduling:** the driver chooses when/how to fulfill requests and when to resume the machine.
- **Separation of concerns:** protocol logic lives in the machine, side effects live in the driver.

---

## Glossary

- **Machine:** the generated Zig type instance that holds persistent state and a resume position.
- **Suspender:** a declared external operation the machine can request.
- **Yield:** a suspension point that returns a request to the driver.
- **Driver:** the outer loop that interprets requests and feeds completions back in.
- **Persistent state:** values stored across yields, declared via `$state`.
- **Procedure:** a reusable coroutine-like routine declared with `$proc`.
- **Submachine:** a named mode/state declared with `$submachine`, entered via `$jump`.

---

## Extended explanation (deep dive)

This section expands on the semantics and mental model.

### The mental model: “a program that pauses”

Write your logic as if it’s normal structured code, but whenever you need something from the outside world (I/O, time, scheduling), you:

1) **yield a request**, and  
2) **wait to be resumed** with the result.

The compiler lowers your direct-style code into a state machine, and `step()` runs it until the next yield/stop boundary.

### What `$yield` really means

A `$yield foo(args...)` is a contract:

- **Machine → Driver:** “Please perform `foo(args...)`.”
- **Driver → Machine:** “Here is the completion value for `foo`.” (or error union value)

This makes external interaction boundaries explicit and easy to trace, log, and test.

### Why `$state` exists

Once you can suspend, “locals on the stack” are no longer a safe assumption. `$state` makes persistence explicit:

- it lives inside the machine struct (`data`)
- it survives yields
- it’s referenced via `${name}` replacement

---

## Implementation details (compiler)

ziggysm compiles in three broad stages:

### 1) Parsing → AST

The input is a mix of:
- plain Zig code (passed through verbatim at top level)
- top-level ziggysm declarations:
  - `$suspender ...;`
  - `$statemachine ... { ... }`
  - `$submachine ... { ... }`
  - `$proc ... { ... }`

Inside blocks, the parser recognizes ziggysm statements:

- `$state name: Type = init;`
- `$yield ...;`
- `$call ...;`
- `$jump ...;`
- `$if (cond) { ... }`
- `$while (cond) { ... }`
- `$loop { ... }`
- `$return ...;` (value only allowed in procedures)
- `$break` / `$continue` (loop control)

Any other text is treated as raw Zig code and preserved as “execute blocks”.

### 2) AST → IR (linear instructions)

The compiler lowers each machine into a small instruction set (a linear program with labels), roughly:

- `EXEC <zig code block>`
- `SETST <state var> = <expr>`
- `YIELD <suspender>(args...) [try] [-> target]`
- `BR <label>` and conditional branches for `$if/$while`
- `CALL <label>` and `RET` for procedures
- `STOP`

Key lowering behaviors:

- `$state` becomes a declared entry in the machine’s `data` struct plus an initializing `SETST`.
- `${name}` inside code blocks becomes an access to a scoped data field (see next section).
- `$jump` becomes a `BR` to the submachine label.
- `$call` becomes `CALL`, and `$return` in procedures becomes `RET`.

### 3) IR → Zig code (the big `switch`)

The generated `step()` function is essentially a **large `switch` over `sm.state`** with a loop label, and a “fallthrough” mechanism using `continue`:

- Each IR label/instruction offset maps to a `State` enum variant.
- Branching sets `sm.state` and `continue`s the switch label.
- Yielding sets `sm.state` to a special “yield resume state” and returns a `Result.<suspender>` request.
- Resuming a yield:
  - verifies the `ReturnValue` tag matches the expected suspender (panic if not)
  - assigns `resume_val.<suspender>` into the chosen target (optionally with `try`)
  - continues running subsequent instructions until the next yield/stop

### Scoped state storage and `${...}` replacement

Internally, `$state` variables are stored in a `data` struct with **scoped names**:

- locals and parameters become `"<scope>:<name>"`
- procedure return slots use just `"<procedure_name>"` as their internal name

`${name}` is replaced at codegen time by:

```zig
sm.data.<scoped_field_name>
```

Replacements are only allowed inside machine/proc/submachine scopes. Plain top-level code does not allow `${...}` replacement.

### Calls, returns, and recursion detection

`$call` is implemented with:

- a per-procedure slot in `branches` storing the return state
- a runtime check that the slot is `null` before setting it (detects recursive re-entry of the same procedure)
- `RET` restores the saved state and clears the slot

This supports structured call/return without an actual stack, but it intentionally prevents recursive calls to the same procedure.

---

## Compiler CLI

The compiler takes exactly one positional input file and an optional output path.

```
ziggysm [options] <input.zigsm>
```

Options:

- `-o, --output <path>`  
  Write the generated Zig code to `<path>`. If omitted, the generated code is written to stdout.

- `-h, --help`  
  Parsed as an option. (Whether it prints usage depends on the argument parsing behavior; the current `main` does not explicitly branch on it.)

Notes:

- The current compiler also prints debug dumps (AST / IR) to stdout during compilation. If you use `--output`, your final Zig code goes to the file, while stdout still contains the debug output.
