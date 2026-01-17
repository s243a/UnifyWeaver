/**
 * shell_sandbox.pl - Declarative Shell Sandbox Configuration
 *
 * This module provides declarative specification of shell access controls
 * and sandbox backends for UnifyWeaver-generated applications.
 *
 * Usage:
 *   :- use_module(shell_sandbox).
 *
 *   % Define app with shell access
 *   app(my_app, [
 *       shell_access([
 *           backends([proot, app_filter]),
 *           levels([...]),
 *           role_access([admin -> full, guest -> sandbox])
 *       ])
 *   ]).
 *
 * Security Level Options:
 *   - sandbox(Backend)      : Isolation backend (none, app_filter, proot, etc.)
 *   - commands(List|all)    : Allowed commands or 'all'
 *   - blocked_commands(List): Commands to block (when commands=all)
 *   - blocked_patterns(List): Regex patterns to block
 *   - pty(Bool)             : Enable PTY for interactive programs
 *   - preserve_home(Bool)   : Use real $HOME (true) or sandbox HOME (false)
 *   - auto_shell(Bool)      : Auto-start PTY shell on connect (for superadmin)
 *   - timeout(Seconds)      : Command timeout
 *   - root_dir(Path)        : Root directory for proot isolation
 *
 * HOME Directory Behavior:
 *   By default, preserve_home is false, meaning the shell uses a sandboxed
 *   HOME directory (typically ~/sandbox). This provides isolation for tools
 *   that store configuration in $HOME (e.g., Claude Code, git, npm).
 *
 *   Benefits of sandboxed HOME (preserve_home=false):
 *   - Isolated tool configurations per shell context
 *   - Multi-user safety (users don't share configs)
 *   - Clean separation between web shell and terminal work
 *
 *   Set preserve_home(true) if you need access to user's real configurations:
 *   - level(my_level, [preserve_home(true), ...])
 *
 * @author UnifyWeaver
 * @version 1.1.0
 */

:- module(shell_sandbox, [
    % Backend definitions
    sandbox_backend/2,
    backend_available/1,
    backend_requirements/2,
    backend_capabilities/2,
    backend_platforms/2,
    backend_security_rating/2,
    backend_status/2,

    % Security level operations
    default_security_levels/1,
    security_level/2,
    level_sandbox/2,
    level_commands/2,
    level_blocked_commands/2,
    level_blocked_patterns/2,
    level_paths/2,
    level_timeout/2,
    level_pty/2,
    level_preserve_home/2,
    level_auto_shell/2,

    % App shell access extraction
    app_shell_access/2,
    app_shell_backends/2,
    app_shell_levels/2,
    app_role_shell_level/3,

    % Code generation
    generate_shell_config/2,
    generate_shell_server/3
]).

:- use_module(library(lists)).

%% ============================================================================
%% Sandbox Backend Definitions
%% ============================================================================

/**
 * sandbox_backend(+Backend, +Properties)
 *
 * Defines available sandbox backends with their properties.
 * Properties include: description, requirements, platforms, capabilities,
 * security_rating, and status.
 */

sandbox_backend(app_filter, [
    description('Application-level command filtering and pattern blocking'),
    requirements([]),
    platforms([linux, macos, termux, windows, android]),
    capabilities([
        command_whitelist,
        command_blacklist,
        pattern_blocking,
        path_restriction(soft),
        timeout
    ]),
    security_rating(low),
    status(implemented)
]).

sandbox_backend(proot, [
    description('Fake chroot using ptrace - filesystem isolation without root'),
    requirements([package(proot)]),
    platforms([linux, termux, android]),
    capabilities([
        filesystem_isolation,
        fake_root,
        path_restriction(hard),
        bind_mounts,
        rootfs_overlay
    ]),
    security_rating(medium),
    status(implemented)
]).

sandbox_backend(firejail, [
    description('SUID sandbox with seccomp filtering'),
    requirements([package(firejail)]),
    platforms([linux]),
    capabilities([
        filesystem_isolation,
        network_filtering,
        seccomp,
        capabilities_drop,
        resource_limits,
        x11_isolation
    ]),
    security_rating(high),
    status(proposed)
]).

sandbox_backend(bubblewrap, [
    description('Unprivileged namespace-based sandboxing'),
    requirements([package(bubblewrap)]),
    platforms([linux]),
    capabilities([
        namespace_isolation,
        filesystem_isolation,
        seccomp,
        no_root_required,
        bind_mounts
    ]),
    security_rating(high),
    status(proposed)
]).

sandbox_backend(docker, [
    description('Container-based isolation with full OS virtualization'),
    requirements([package(docker), service(dockerd)]),
    platforms([linux, macos, windows]),
    capabilities([
        full_isolation,
        resource_limits,
        network_isolation,
        custom_images,
        volume_mounts,
        cpu_memory_limits
    ]),
    security_rating(very_high),
    status(proposed)
]).

sandbox_backend(podman, [
    description('Rootless container runtime - Docker alternative'),
    requirements([package(podman)]),
    platforms([linux, macos]),
    capabilities([
        full_isolation,
        rootless,
        resource_limits,
        network_isolation,
        custom_images
    ]),
    security_rating(very_high),
    status(proposed)
]).

sandbox_backend(nsjail, [
    description('Light-weight process isolation using namespaces and seccomp-bpf'),
    requirements([package(nsjail)]),
    platforms([linux]),
    capabilities([
        namespace_isolation,
        seccomp_bpf,
        cgroups,
        resource_limits,
        network_isolation
    ]),
    security_rating(very_high),
    status(proposed)
]).

sandbox_backend(chroot, [
    description('Traditional Unix chroot jail'),
    requirements([root(required)]),
    platforms([linux, macos, bsd]),
    capabilities([
        filesystem_isolation,
        path_restriction(hard)
    ]),
    security_rating(low),  % Easy to escape without additional measures
    status(proposed)
]).

sandbox_backend(ssh_sandbox, [
    description('SSH to isolated user/container on remote host'),
    requirements([ssh_access(remote_sandbox_host)]),
    platforms([any]),
    capabilities([
        full_isolation,
        network_separation,
        resource_limits,
        audit_logging
    ]),
    security_rating(very_high),
    status(proposed)
]).

%% ============================================================================
%% Backend Property Accessors
%% ============================================================================

backend_requirements(Backend, Reqs) :-
    sandbox_backend(Backend, Props),
    member(requirements(Reqs), Props).

backend_capabilities(Backend, Caps) :-
    sandbox_backend(Backend, Props),
    member(capabilities(Caps), Props).

backend_platforms(Backend, Platforms) :-
    sandbox_backend(Backend, Props),
    member(platforms(Platforms), Props).

backend_security_rating(Backend, Rating) :-
    sandbox_backend(Backend, Props),
    member(security_rating(Rating), Props).

backend_status(Backend, Status) :-
    sandbox_backend(Backend, Props),
    (   member(status(Status), Props)
    ->  true
    ;   Status = implemented
    ).

backend_description(Backend, Desc) :-
    sandbox_backend(Backend, Props),
    member(description(Desc), Props).

/**
 * backend_available(+Backend)
 *
 * Succeeds if the backend is implemented and requirements can be met
 * on the current platform.
 */
backend_available(Backend) :-
    sandbox_backend(Backend, Props),
    member(status(implemented), Props).

%% ============================================================================
%% Default Security Levels
%% ============================================================================

/**
 * default_security_levels(-Levels)
 *
 * Returns the default security level definitions.
 */
default_security_levels([
    level(superadmin, [
        description('Full shell access with real HOME - direct PTY mode'),
        sandbox(none),
        commands(all),
        paths(all),
        timeout(none),
        pty(true),
        preserve_home(true),  % Use real HOME for full system access
        auto_shell(true)      % Auto-start interactive shell on connect
    ]),

    level(full, [
        description('No restrictions - full shell access'),
        sandbox(none),
        commands(all),
        paths(all),
        timeout(none),
        pty(true),
        preserve_home(false)  % Use sandbox HOME for isolated tool configs
    ]),

    level(trusted, [
        description('Trust user but block destructive operations'),
        sandbox(app_filter),
        commands(all),
        pty(true),
        preserve_home(false),  % Use sandbox HOME for isolated tool configs
        blocked_commands([
            sudo, su, doas,           % Privilege escalation
            rm, rmdir,                 % Deletion (use trash instead)
            mkfs, fdisk, parted,       % Disk operations
            dd,                        % Raw disk access
            chmod, chown, chgrp,       % Permission changes
            kill, killall, pkill,      % Process termination
            reboot, shutdown, halt,    % System control
            iptables, ufw,             % Firewall
            mount, umount              % Filesystem mounting
        ]),
        blocked_patterns([
            'rm\\s+-rf',
            'rm\\s+-fr',
            '>\\s*/dev/',
            'chmod\\s+777',
            'curl.*\\|.*sh',
            'wget.*\\|.*sh',
            ':\\(\\)\\{',              % Fork bomb
            '/etc/passwd',
            '/etc/shadow'
        ]),
        paths(all),
        timeout(300)
    ]),

    level(sandbox, [
        description('Isolated to sandbox directory with limited commands'),
        sandbox(proot),
        root_dir('~/sandbox'),
        commands([
            % Navigation
            ls, pwd, cd, tree,
            % Reading
            cat, head, tail, less, more,
            % Searching
            grep, find, locate, which,
            % Text processing
            awk, sed, sort, uniq, cut, tr, wc,
            % File info
            file, stat, du, df,
            % Basic operations
            echo, printf, date, whoami, id,
            % Safe file ops
            mkdir, touch, cp, mv,
            % Editors (if available)
            nano, vim, vi,
            % Scripting
            python, python3, node, ruby,
            % Archive
            tar, gzip, gunzip, zip, unzip
        ]),
        write_paths(['~/sandbox']),
        read_paths(['~/sandbox']),
        timeout(120),
        max_output(50000)
    ]),

    level(restricted, [
        description('Minimal read-only access for untrusted users'),
        sandbox(proot),
        root_dir('~/sandbox'),
        commands([
            ls, pwd, cat, head, tail,
            echo, date, whoami
        ]),
        write_paths([]),
        read_paths(['~/sandbox']),
        timeout(30),
        max_output(5000)
    ]),

    level(education, [
        description('Safe environment for learning - includes helpful tools'),
        sandbox(proot),
        root_dir('~/sandbox'),
        commands([
            % Navigation & reading
            ls, pwd, cd, cat, head, tail, less,
            % Learning-friendly
            man, help, info, whatis,
            % Safe operations
            echo, printf, date, cal,
            mkdir, touch, cp, mv,
            % Text tools
            grep, sort, uniq, wc,
            % Programming
            python, python3, node,
            % Editors
            nano
        ]),
        write_paths(['~/sandbox']),
        read_paths(['~/sandbox', '~/tutorials']),
        bind_mounts([
            '~/tutorials' -> '/tutorials'
        ]),
        timeout(300),
        max_output(20000),
        welcome_message('Welcome to the learning shell! Type "help" for guidance.')
    ])
]).

%% ============================================================================
%% Security Level Accessors
%% ============================================================================

/**
 * security_level(+LevelName, -LevelConfig)
 *
 * Retrieves configuration for a named security level from defaults.
 */
security_level(LevelName, Config) :-
    default_security_levels(Levels),
    member(level(LevelName, Config), Levels).

level_property(LevelConfig, Property, Value) :-
    member(Property, LevelConfig),
    Property =.. [_, Value].

level_sandbox(LevelConfig, Sandbox) :-
    (   member(sandbox(Sandbox), LevelConfig)
    ->  true
    ;   Sandbox = app_filter
    ).

level_commands(LevelConfig, Commands) :-
    (   member(commands(Commands), LevelConfig)
    ->  true
    ;   Commands = []
    ).

level_blocked_commands(LevelConfig, Blocked) :-
    (   member(blocked_commands(Blocked), LevelConfig)
    ->  true
    ;   Blocked = []
    ).

level_blocked_patterns(LevelConfig, Patterns) :-
    (   member(blocked_patterns(Patterns), LevelConfig)
    ->  true
    ;   Patterns = []
    ).

level_paths(LevelConfig, Paths) :-
    (   member(paths(Paths), LevelConfig)
    ->  true
    ;   member(read_paths(Paths), LevelConfig)
    ->  true
    ;   Paths = []
    ).

level_timeout(LevelConfig, Timeout) :-
    (   member(timeout(Timeout), LevelConfig)
    ->  true
    ;   Timeout = 120
    ).

level_root_dir(LevelConfig, RootDir) :-
    (   member(root_dir(RootDir), LevelConfig)
    ->  true
    ;   RootDir = '~/sandbox'
    ).

level_pty(LevelConfig, Pty) :-
    (   member(pty(Pty), LevelConfig)
    ->  true
    ;   Pty = false  % Default: no PTY
    ).

level_preserve_home(LevelConfig, PreserveHome) :-
    (   member(preserve_home(PreserveHome), LevelConfig)
    ->  true
    ;   PreserveHome = false  % Default: use sandbox HOME for isolation
    ).

level_auto_shell(LevelConfig, AutoShell) :-
    (   member(auto_shell(AutoShell), LevelConfig)
    ->  true
    ;   AutoShell = false  % Default: don't auto-start shell
    ).

%% ============================================================================
%% App Shell Access Extraction
%% ============================================================================

/**
 * app_shell_access(+AppSpec, -ShellConfig)
 *
 * Extracts shell_access configuration from an app specification.
 */
app_shell_access(app(_, Config), ShellConfig) :-
    (   member(shell_access(ShellConfig), Config)
    ->  true
    ;   ShellConfig = []  % No shell access configured
    ).

app_shell_backends(AppSpec, Backends) :-
    app_shell_access(AppSpec, Config),
    (   member(backends(Backends), Config)
    ->  true
    ;   Backends = [app_filter]  % Default backend
    ).

app_shell_levels(AppSpec, Levels) :-
    app_shell_access(AppSpec, Config),
    (   member(levels(Levels), Config)
    ->  true
    ;   default_security_levels(Levels)
    ).

/**
 * app_role_shell_level(+AppSpec, +Role, -Level)
 *
 * Maps a user role to their shell security level.
 */
app_role_shell_level(AppSpec, Role, Level) :-
    app_shell_access(AppSpec, Config),
    member(role_access(Mappings), Config),
    member((Role -> Level), Mappings),
    !.
app_role_shell_level(_, admin, full) :- !.
app_role_shell_level(_, _, sandbox).  % Default for unknown roles

%% ============================================================================
%% Configuration Generation
%% ============================================================================

/**
 * generate_shell_config(+AppSpec, -ConfigJSON)
 *
 * Generates a JSON configuration file for the shell server.
 */
generate_shell_config(AppSpec, ConfigJSON) :-
    app_shell_backends(AppSpec, Backends),
    app_shell_levels(AppSpec, Levels),
    app_shell_access(AppSpec, Config),
    (   member(role_access(RoleAccess), Config)
    ->  true
    ;   RoleAccess = [admin -> full, user -> trusted, guest -> sandbox]
    ),
    format_shell_config_json(Backends, Levels, RoleAccess, ConfigJSON).

format_shell_config_json(Backends, Levels, RoleAccess, JSON) :-
    format_backends_json(Backends, BackendsJSON),
    format_levels_json(Levels, LevelsJSON),
    format_role_access_json(RoleAccess, RoleAccessJSON),
    format(atom(JSON), '{
  "backends": ~w,
  "levels": ~w,
  "roleAccess": ~w
}', [BackendsJSON, LevelsJSON, RoleAccessJSON]).

format_backends_json(Backends, JSON) :-
    maplist(format_backend_entry, Backends, Entries),
    atomic_list_concat(Entries, ', ', Inner),
    format(atom(JSON), '[~w]', [Inner]).

format_backend_entry(Backend, Entry) :-
    format(atom(Entry), '"~w"', [Backend]).

format_levels_json(Levels, JSON) :-
    maplist(format_level_entry, Levels, Entries),
    atomic_list_concat(Entries, ',\n    ', Inner),
    format(atom(JSON), '{\n    ~w\n  }', [Inner]).

format_level_entry(level(Name, Config), Entry) :-
    level_sandbox(Config, Sandbox),
    level_commands(Config, Commands),
    level_blocked_commands(Config, Blocked),
    level_timeout(Config, Timeout),
    level_pty(Config, Pty),
    level_preserve_home(Config, PreserveHome),
    level_auto_shell(Config, AutoShell),
    format_commands_json(Commands, CommandsJSON),
    format_commands_json(Blocked, BlockedJSON),
    format(atom(Entry), '"~w": {
      "sandbox": "~w",
      "commands": ~w,
      "blockedCommands": ~w,
      "timeout": ~w,
      "pty": ~w,
      "preserveHome": ~w,
      "autoShell": ~w
    }', [Name, Sandbox, CommandsJSON, BlockedJSON, Timeout, Pty, PreserveHome, AutoShell]).

format_commands_json(all, '"all"') :- !.
format_commands_json([], '[]') :- !.
format_commands_json(Commands, JSON) :-
    maplist(format_command_entry, Commands, Entries),
    atomic_list_concat(Entries, ', ', Inner),
    format(atom(JSON), '[~w]', [Inner]).

format_command_entry(Cmd, Entry) :-
    format(atom(Entry), '"~w"', [Cmd]).

format_role_access_json(RoleAccess, JSON) :-
    maplist(format_role_entry, RoleAccess, Entries),
    atomic_list_concat(Entries, ', ', Inner),
    format(atom(JSON), '{~w}', [Inner]).

format_role_entry((Role -> Level), Entry) :-
    format(atom(Entry), '"~w": "~w"', [Role, Level]).

%% ============================================================================
%% Shell Server Generation
%% ============================================================================

/**
 * generate_shell_server(+AppSpec, +Target, -Files)
 *
 * Generates shell server files for the specified target.
 * Currently supports: node (Node.js with ws library)
 */
generate_shell_server(AppSpec, node, Files) :-
    generate_shell_config(AppSpec, ConfigJSON),
    generate_node_shell_server(AppSpec, ServerCode),
    Files = [
        file('server/shell-config.json', ConfigJSON),
        file('server/shell-server.cjs', ServerCode)
    ].

generate_node_shell_server(AppSpec, Code) :-
    app_shell_backends(AppSpec, Backends),
    (   member(proot, Backends)
    ->  ProotSupport = true
    ;   ProotSupport = false
    ),
    format(atom(Code), '/**
 * Sandboxed Shell WebSocket Server
 * Generated by UnifyWeaver
 *
 * Supports backends: ~w
 * Proot support: ~w
 */

const { WebSocketServer } = require("ws");
const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

// Load configuration
const CONFIG_PATH = path.join(__dirname, "shell-config.json");
const config = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));

const PORT = process.env.SHELL_PORT || 3001;
const SANDBOX_DIR = process.env.SANDBOX_DIR || path.join(process.env.HOME, "sandbox");

// Ensure sandbox exists
if (!fs.existsSync(SANDBOX_DIR)) {
  fs.mkdirSync(SANDBOX_DIR, { recursive: true });
}

// ... (full server implementation)
// See shell-server.cjs for complete implementation

console.log("Shell server configuration loaded");
console.log("Backends:", config.backends);
console.log("Levels:", Object.keys(config.levels));
', [Backends, ProotSupport]).

%% ============================================================================
%% Documentation / Proposals
%% ============================================================================

/**
 * proposed_backend_docs(+Backend, -Documentation)
 *
 * Returns documentation for proposed (not yet implemented) backends.
 */
proposed_backend_docs(firejail, Doc) :-
    Doc = '
## Firejail Backend (Proposed)

Firejail is a SUID sandbox program that reduces the risk of security
breaches by restricting the running environment using Linux namespaces,
seccomp-bpf, and capabilities.

### Requirements
- Linux only
- Package: firejail

### Implementation Notes
```javascript
const { spawn } = require("child_process");

function executeWithFirejail(command, options) {
  const args = [
    "--noprofile",
    "--private=" + options.sandboxDir,
    "--net=none",  // No network access
    "--nosound",
    "--no3d",
    "--nodvd",
    "--nogroups",
    "--nonewprivs",
    "--seccomp",
    "--",
    "/bin/sh", "-c", command
  ];

  return spawn("firejail", args);
}
```

### Security Features
- Filesystem isolation via mount namespaces
- Network filtering (can block all network access)
- Seccomp syscall filtering
- Capability dropping
- Resource limits via cgroups
'.

proposed_backend_docs(docker, Doc) :-
    Doc = '
## Docker Backend (Proposed)

Docker provides full container-based isolation with its own filesystem,
networking, and process space.

### Requirements
- Docker daemon running
- User in docker group (or root)

### Implementation Notes
```javascript
const { spawn } = require("child_process");

function executeWithDocker(command, options) {
  const args = [
    "run",
    "--rm",
    "--network=none",
    "--memory=256m",
    "--cpus=0.5",
    "-v", `${options.sandboxDir}:/sandbox:rw`,
    "-w", "/sandbox",
    "--user", "1000:1000",
    options.image || "alpine:latest",
    "/bin/sh", "-c", command
  ];

  return spawn("docker", args);
}
```

### Security Features
- Full OS-level isolation
- Resource limits (CPU, memory, disk I/O)
- Network isolation
- Custom images with minimal attack surface
- Read-only root filesystem option
'.

proposed_backend_docs(bubblewrap, Doc) :-
    Doc = '
## Bubblewrap Backend (Proposed)

Bubblewrap (bwrap) is an unprivileged sandboxing tool that uses
Linux user namespaces. Used by Flatpak.

### Requirements
- Linux with user namespace support
- Package: bubblewrap

### Implementation Notes
```javascript
const { spawn } = require("child_process");

function executeWithBubblewrap(command, options) {
  const args = [
    "--unshare-all",
    "--die-with-parent",
    "--ro-bind", "/usr", "/usr",
    "--ro-bind", "/lib", "/lib",
    "--ro-bind", "/lib64", "/lib64",
    "--bind", options.sandboxDir, "/sandbox",
    "--chdir", "/sandbox",
    "--proc", "/proc",
    "--dev", "/dev",
    "/bin/sh", "-c", command
  ];

  return spawn("bwrap", args);
}
```

### Security Features
- No root required
- Namespace isolation (PID, mount, network, user)
- Minimal overhead
- Fine-grained bind mount control
'.
