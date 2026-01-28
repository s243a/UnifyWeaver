# Proposal: Flutter Generator Testing and Fixes

## Overview

The Flutter generator (`src/unifyweaver/ui/flutter_generator.pl`) produces Dart/Flutter code from UI specifications, but has not been tested against a real Flutter environment. This proposal covers testing the generator and fixing issues to achieve feature parity with the recently-fixed React generator.

## Current State

The generated Flutter code in `examples/flutter-cli/` has several issues:

1. **Undefined variables** - References `browse.path`, `browse.entries` without state class definitions
2. **Invalid Dart syntax** - Uses `...list.map()` spread incorrectly
3. **Missing state management** - No reactive state or API integration
4. **Undefined handlers** - References `navigate_up`, `view_file`, etc. without implementations
5. **No API client** - Unlike React/Vue versions, no HTTP client code generated

These are similar to the issues found and fixed in the React generator.

## Test Environment Options

### Option 1: Raw Termux (Recommended for Web)

| Aspect | Details |
|--------|---------|
| **Setup** | Download Flutter SDK linux-arm64, add to PATH |
| **Pros** | Simplest setup, no proot overhead, direct filesystem access |
| **Cons** | Some packages may have Termux-specific issues |
| **Best for** | Flutter Web development/testing |

```bash
curl -LO https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.27.4-stable.tar.xz
tar xf flutter_linux_3.27.4-stable.tar.xz
export PATH="$HOME/flutter/bin:$PATH"
flutter config --enable-web
```

### Option 2: Proot Debian

| Aspect | Details |
|--------|---------|
| **Setup** | `proot-distro login debian`, install Flutter inside |
| **Pros** | Standard Linux environment, better compatibility |
| **Cons** | Proot overhead, filesystem path translation complexity |
| **Best for** | If raw Termux has issues, or for Android APK builds |

### Option 3: Proot Arch (FlutterArch)

| Aspect | Details |
|--------|---------|
| **Setup** | [FlutterArch](https://github.com/bdloser404/FlutterArch) script installs Flutter in Arch proot |
| **Pros** | Automated setup, includes Neovim config for Flutter dev |
| **Cons** | Requires separate Arch proot install, script maintained by third party |
| **Best for** | Users who prefer Arch, want turnkey Flutter setup |

### Option 4: Windows/WSL (Desktop)

| Aspect | Details |
|--------|---------|
| **Setup** | Standard Flutter install on Windows or WSL |
| **Pros** | Full Flutter support, Android Studio integration |
| **Cons** | Can't develop on mobile, requires separate machine |
| **Best for** | Final APK builds, comprehensive testing |

### Option 5: Flutter Web Only (Minimal)

| Aspect | Details |
|--------|---------|
| **Setup** | Flutter SDK + browser only |
| **Pros** | No Android SDK needed, quick iteration |
| **Cons** | Can't test mobile-specific features |
| **Best for** | Generator validation, UI prototyping |

### Recommendation

**Start with Option 1 (Raw Termux) + Option 5 (Web only)**. This provides:
- Fastest iteration cycle
- Minimal setup friction
- Sufficient coverage for generator validation

Fall back to Option 2 (Debian) or Option 3 (Arch) if raw Termux has compatibility issues. Use Option 4 (Windows/WSL) for final Android builds.

## Philosophy: Why Flutter Web is Sufficient for Development

### Cross-Platform Code Identity

Flutter's core value proposition is **one codebase, multiple targets**. The Dart code, widget tree, and state management are identical whether targeting:
- Web (HTML/JS/Canvas)
- Android (APK)
- iOS (IPA)
- Desktop (Windows/macOS/Linux)

Only the rendering backend differs. The Flutter framework abstracts this entirely.

### Generator Validation Scope

Our goal is to validate that the **generator produces correct Dart/Flutter code**:
- Valid Dart syntax
- Proper widget composition
- Correct state management patterns
- Working event handlers

These are **target-agnostic**. If the code compiles and runs on Web, it will compile and run on Android. The generator doesn't produce platform-specific code.

### Development Workflow Benefits

| Web | Native APK |
|-----|------------|
| `flutter build web` (~10s) | `flutter build apk` (~60s+) |
| Refresh browser | Reinstall APK |
| No signing required | Needs keystore |
| Debug in browser DevTools | Needs ADB/device |
| No SSL certificate issues with localhost | App can bypass cert validation |

For rapid iteration during generator development, Web is significantly faster.

### Chrome Android API Support

Chrome on Android provides access to many device APIs, reducing the gap between web and native:

**Supported in Chrome Android:**
| API | Description | Notes |
|-----|-------------|-------|
| File System Access | `showOpenFilePicker()` - native file browser | Used in our React/Vue upload |
| Media Capture | Camera/microphone via `getUserMedia()` | Full support |
| Geolocation | GPS access | Requires HTTPS |
| Web Share | Native share sheet | `navigator.share()` |
| Device Motion | Accelerometer, gyroscope | `DeviceMotionEvent` |
| Device Orientation | Compass, tilt | `DeviceOrientationEvent` |
| Vibration | Haptic feedback | `navigator.vibrate()` |
| Clipboard | Read/write clipboard | Requires permission |
| Web Bluetooth | BLE device access | Experimental |
| Web NFC | NFC tag read/write | Limited device support |
| Contact Picker | Access contacts | `navigator.contacts.select()` |
| Push Notifications | Background notifications | Via Service Worker |
| WebSocket | Real-time communication | Used in our shell feature |

**NOT supported (require native app):**
| API | Why native-only |
|-----|-----------------|
| Background execution | Browser tabs suspend |
| SMS/Phone calls | Security restriction |
| Full filesystem access | Sandboxed to picker |
| System settings | OS-level access |
| Other app integration | Intent system |
| Widgets | Android launcher feature |
| Always-on services | Background limits |

### When Native Matters

Native (APK) builds become relevant for:
- Background execution (services that run when app is closed)
- System-level integration (SMS, calls, intents)
- Home screen widgets
- App store distribution
- Performance-critical graphics (games)

For our HTTP CLI app, Chrome Android APIs cover everything we need:
- File picking ✅ (File System Access API)
- WebSocket shell ✅
- HTTPS API calls ✅
- File download ✅

None of the native-only features apply to generator validation.

## Implementation Plan

### Phase 1: Environment Setup
- Install Flutter SDK on Termux
- Enable web support
- Verify with `flutter doctor`

### Phase 2: Analyze Generator Issues
- Compare `flutter_generator.pl` patterns with fixed `react_generator.pl`
- Identify Dart-specific syntax requirements
- Document required helper predicates

### Phase 3: Fix Generator

**Syntax Fixes:**
- Convert variable references to Dart syntax (`browse.path` → `_browse['path']` or proper state class)
- Fix list rendering (spread operator → `ListView.builder` or `.map().toList()`)
- Add null safety operators where needed

**State Management:**
- Generate proper `StatefulWidget` with `State` class
- Add state variables with correct types
- Generate `setState()` calls for reactivity

**API Integration:**
- Add HTTP client (using `http` package or `dio`)
- Generate async methods for API calls
- Handle loading/error states

**Handler Generation:**
- Generate method stubs with proper signatures
- Wire up `onPressed`, `onTap` callbacks correctly

### Phase 4: Test Full App
- Generate complete HTTP CLI app (like React version)
- Build with `flutter build web`
- Serve and test in browser
- Verify all tabs work: Browse, Upload, Cat, Shell

### Phase 5: Documentation
- Update `examples/flutter-cli/README.md`
- Add to feature parity matrix in `FUTURE_WORK.md`
- Create PR with test plan

## Files to Modify

| File | Changes |
|------|---------|
| `src/unifyweaver/ui/flutter_generator.pl` | Core syntax and pattern fixes |
| `src/unifyweaver/ui/project_scaffold.pl` | Flutter project scaffolding improvements |
| `examples/flutter-cli/lib/app.dart` | Regenerate with fixes |
| `examples/flutter-cli/pubspec.yaml` | Add http/dio dependency |
| `examples/flutter-cli/README.md` | Setup and usage docs |

## Success Criteria

1. `flutter analyze` passes with no errors
2. `flutter build web` succeeds
3. App loads in browser without console errors
4. All tabs functional (Browse, Upload, Cat, Shell)
5. API integration works with HTTP CLI server
6. Feature parity with React HTTP CLI app

## Comparison: Generator Feature Parity

| Feature | Vue | React | Flutter (Target) |
|---------|-----|-------|------------------|
| State management | ✅ reactive() | ✅ useState | ⬜ StatefulWidget |
| API client | ✅ fetch | ✅ fetch | ⬜ http/dio |
| Root selector | ✅ | ✅ | ⬜ |
| File upload | ✅ FSA API | ✅ FSA API | ⬜ file_picker |
| WebSocket shell | ✅ | ✅ | ⬜ web_socket_channel |
| HTTPS support | ✅ | ✅ | ⬜ |

## Timeline

1. Environment setup: User-driven
2. Generator fixes: 1-2 sessions
3. Testing and iteration: 1 session
4. Documentation and PR: 1 session

## References

- [Flutter Web documentation](https://docs.flutter.dev/platform-integration/web)
- [termux-flutter project](https://github.com/mumumusuc/termux-flutter)
- [Flutter SDK archive](https://docs.flutter.dev/install/archive)
- Related PR: `PR_REACT_HTTP_CLI.md` (similar fixes for React)
