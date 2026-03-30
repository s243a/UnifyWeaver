# Android Runtime Investigation — Termux DEX Execution

## Date: 2026-03-29

## Goal

Determine if we can execute DEX bytecode on Termux (Android 14,
Samsung S24, aarch64) without `adb`, to validate a future smali
assembly target for UnifyWeaver.

## What Works

### DEX compilation (dx)

`dx` is available via `pkg install dx`. Successfully converts
Java 8 .class files to DEX format:

```bash
javac -source 1.8 -target 1.8 HelloWorld.java
dx --dex --output=hello.dex HelloWorld.class
# hello.dex created (752 bytes)
```

**Caveat:** `dx` requires Java 8 class file format. Java 21
(Termux default) produces class version 65.0 which `dx` rejects.
Use `-source 1.8 -target 1.8` flags, or use `d8` (not currently
in Termux packages but available as an Android SDK tool).

### smali Python library

`pip install smali` installs a Python library for programmatic
smali manipulation. It's NOT a CLI assembler — it's a parser/AST
library. Could potentially be used for validation.

### BOOTCLASSPATH

The system `BOOTCLASSPATH` environment variable is set and points
to the full Android framework JARs (core-oj.jar, framework.jar,
etc.). This is available for future approaches.

## What Fails

### dalvikvm

The Termux wrapper at `/data/data/com.termux/files/usr/bin/dalvikvm`
sets up `ANDROID_DATA` and calls the system dalvikvm. Execution
crashes immediately with:

```
runtime.cc:717] at java.lang.ClassLoader.createSystemClassLoader()
Aborted (exit code 134)
```

**Root cause:** On Android 14, ART expects to be launched via
Zygote with full framework services initialized. A raw `dalvikvm`
process can't create the system class loader because it lacks the
framework context that Zygote provides.

Even with explicit `-Xbootclasspath:$BOOTCLASSPATH`, the crash
persists. The issue is not about finding boot jars but about the
process context (SELinux labels, Zygote-inherited state, etc.).

### app_process

`/system/bin/app_process` is the framework process launcher:

```bash
app_process -Djava.class.path=$HOME/dex_test/hello.dex /system/bin HelloWorld
```

- With `/system/bin` as the base dir: exits 0 but no output,
  then aborts on redirect
- With `/` as base dir: aborts immediately (exit 134)

Same root cause as dalvikvm — the process context isn't right
for Termux's unprivileged environment on Android 14.

## Available Tools (Termux)

| Tool | Package | Status |
|------|---------|--------|
| `dalvikvm` | Built-in wrapper | Crashes on Android 14 |
| `dx` | `pkg install dx` | Works (Java 8 → DEX) |
| `javac` | `pkg install openjdk-21` | Works |
| `java` (JVM) | `pkg install openjdk-21` | Works (not Dalvik) |
| `app_process` | System binary | Crashes from Termux |
| `smali` (pip) | `pip install smali` | Python library only, no CLI |
| `smali`/`baksmali` (CLI) | Not in Termux packages | Would need manual install as JAR |

## Possible Future Approaches

### 1. adb shell

Running via `adb shell` from another device/session provides a
different process context that may have the permissions needed.
`adb shell dalvikvm` or `adb shell app_process` historically
works on many Android versions. **Downside:** requires a second
device or USB connection, awkward for automated testing.

### 2. Rooted device

With root access, SELinux can be set to permissive mode or the
process can be launched with the correct security context:

```bash
su -c "dalvikvm -cp /path/to/hello.dex HelloWorld"
# or
su -c "setenforce 0 && dalvikvm ..."
```

### 3. Validate without execution

Generate smali assembly text and validate syntax/structure
without running it:

- Use the `smali` pip library for AST validation
- Download `baksmali.jar` and `smali.jar` from the Android SDK
  tools (run via `java -jar smali.jar assemble file.smali`)
- Round-trip: smali → DEX → baksmali → verify

This is the most practical approach for CI/testing. The generated
code's *logic* can be validated via the JVM path (Jamaica/Krakatau
targets produce equivalent bytecode that runs on OpenJDK).

### 4. Termux:API or Phantom Process

Android 12+ has "phantom process killing" which may affect
long-running dalvikvm processes. Disabling it requires:

```bash
adb shell "settings put global settings_enable_monitor_phantom_procs false"
```

This might not fix the class loader issue but could help with
other runtime problems.

### 5. d8 instead of dx

Google's `d8` is the modern replacement for `dx` with better
support for newer class file formats. Not currently in Termux
packages but could be installed as a standalone JAR:

```bash
# Download d8 from Android SDK build-tools
java -jar d8.jar --output hello.dex HelloWorld.class
```

## Recommendation

For the smali target, pursue **approach 3 (validate without
execution)** as the primary testing strategy:

1. Generate smali text from UnifyWeaver
2. Validate via `java -jar smali.jar assemble` (assemble to DEX)
3. Test logic equivalence via Jamaica/Krakatau JVM targets
4. Optionally validate on a rooted device or via adb

The smali target itself would follow the same architecture as
ILAsm — shared bytecode module (`dex_bytecode.pl`), compile_expression
hooks, recursion patterns, composable TC templates. The testing
strategy is the only difference from other assembly targets.

## Environment Details

- Device: Samsung S24
- Android: 14 (kernel 6.1.128-android14-11)
- Termux: current stable
- Java: OpenJDK 21
- Architecture: aarch64
