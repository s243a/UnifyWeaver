/**
 * auth_backends.pl - Declarative Authentication Backend Configuration
 *
 * This module provides declarative specification of authentication backends
 * for UnifyWeaver-generated applications.
 *
 * Usage:
 *   :- use_module(auth_backends).
 *
 *   % Define app with auth configuration
 *   app(my_app, [
 *       auth([
 *           backend(text_file),           % Storage backend
 *           password_hash(bcrypt),         % Hashing algorithm
 *           token_type(jwt),               % Token type
 *           users_file('users.txt'),       % Backend-specific config
 *           encryption(none)               % File encryption (for text backends)
 *       ])
 *   ]).
 *
 * Backend Options:
 *   - backend(Type)         : Storage backend (mock, text_file, sqlite, etc.)
 *   - password_hash(Algo)   : Password hashing (plain, bcrypt, argon2, scrypt)
 *   - token_type(Type)      : Token type (jwt, session, api_key)
 *   - encryption(Type)      : File encryption (none, aes256) - for file backends
 *   - users_file(Path)      : Path to users file/database
 *   - session_duration(Sec) : Session/token duration in seconds
 *
 * @author UnifyWeaver
 * @version 1.0.0
 */

:- module(auth_backends, [
    % Backend definitions
    auth_backend/2,
    backend_available/1,
    backend_requirements/2,
    backend_capabilities/2,
    backend_status/2,

    % Password hashing
    hash_algorithm/2,
    hash_available/1,

    % Token types
    token_type/2,

    % App auth extraction
    app_auth_config/2,
    app_auth_backend/2,
    app_auth_hash/2,
    app_auth_token_type/2,

    % Code generation
    generate_auth_config/2,
    generate_auth_server/3
]).

:- use_module(library(lists)).

%% ============================================================================
%% Authentication Backend Definitions
%% ============================================================================

/**
 * auth_backend(+Backend, +Properties)
 *
 * Defines available authentication backends with their properties.
 */

auth_backend(mock, [
    description('Client-side mock authentication for development'),
    requirements([]),
    capabilities([
        no_persistence,
        fast_setup,
        testing_only
    ]),
    platforms([all]),
    status(implemented)
]).

auth_backend(text_file, [
    description('Simple text file storage with one user per line'),
    requirements([]),
    capabilities([
        persistence,
        simple_setup,
        portable,
        human_readable
    ]),
    format('username:password_hash:roles:permissions'),
    platforms([all]),
    status(implemented)
]).

auth_backend(text_file_encrypted, [
    description('Encrypted text file storage'),
    requirements([encryption_key]),
    capabilities([
        persistence,
        encrypted_at_rest,
        portable
    ]),
    platforms([all]),
    status(proposed)
]).

auth_backend(sqlite, [
    description('SQLite database storage'),
    requirements([package(sqlite3)]),
    capabilities([
        persistence,
        queryable,
        transactions,
        portable
    ]),
    platforms([all]),
    status(proposed)
]).

auth_backend(postgresql, [
    description('PostgreSQL database storage'),
    requirements([package(postgresql), service(postgresql)]),
    capabilities([
        persistence,
        queryable,
        transactions,
        scalable,
        concurrent
    ]),
    platforms([linux, macos, windows]),
    status(proposed)
]).

auth_backend(mongodb, [
    description('MongoDB document storage'),
    requirements([package(mongodb), service(mongod)]),
    capabilities([
        persistence,
        document_store,
        scalable,
        flexible_schema
    ]),
    platforms([linux, macos, windows]),
    status(proposed)
]).

auth_backend(ldap, [
    description('LDAP/Active Directory authentication'),
    requirements([ldap_server]),
    capabilities([
        external_auth,
        enterprise,
        centralized
    ]),
    platforms([all]),
    status(proposed)
]).

auth_backend(oauth2, [
    description('OAuth2 provider integration'),
    requirements([oauth_provider]),
    capabilities([
        external_auth,
        social_login,
        delegated
    ]),
    providers([google, github, microsoft, facebook]),
    platforms([all]),
    status(proposed)
]).

%% ============================================================================
%% Password Hashing Algorithms
%% ============================================================================

hash_algorithm(plain, [
    description('No hashing - plain text (development only)'),
    security(none),
    status(implemented)
]).

hash_algorithm(bcrypt, [
    description('bcrypt adaptive hashing'),
    security(high),
    work_factor(12),
    status(implemented)
]).

hash_algorithm(argon2, [
    description('Argon2id memory-hard hashing'),
    security(very_high),
    status(proposed)
]).

hash_algorithm(scrypt, [
    description('scrypt memory-hard hashing'),
    security(high),
    status(proposed)
]).

hash_algorithm(sha256, [
    description('SHA-256 with salt'),
    security(medium),
    status(implemented)
]).

hash_available(Algo) :-
    hash_algorithm(Algo, Props),
    member(status(implemented), Props).

%% ============================================================================
%% Token Types
%% ============================================================================

token_type(jwt, [
    description('JSON Web Token - stateless'),
    capabilities([stateless, expiry, claims]),
    status(implemented)
]).

token_type(session, [
    description('Server-side session with cookie'),
    capabilities([stateful, revocable, server_storage]),
    status(proposed)
]).

token_type(api_key, [
    description('Static API key authentication'),
    capabilities([simple, no_expiry, machine_to_machine]),
    status(proposed)
]).

%% ============================================================================
%% Backend Property Accessors
%% ============================================================================

backend_requirements(Backend, Reqs) :-
    auth_backend(Backend, Props),
    member(requirements(Reqs), Props).

backend_capabilities(Backend, Caps) :-
    auth_backend(Backend, Props),
    member(capabilities(Caps), Props).

backend_status(Backend, Status) :-
    auth_backend(Backend, Props),
    (   member(status(Status), Props)
    ->  true
    ;   Status = implemented
    ).

backend_available(Backend) :-
    auth_backend(Backend, Props),
    member(status(implemented), Props).

%% ============================================================================
%% App Auth Extraction
%% ============================================================================

/**
 * app_auth_config(+AppSpec, -AuthConfig)
 *
 * Extracts auth configuration from an app specification.
 */
app_auth_config(app(_, Config), AuthConfig) :-
    (   member(auth(AuthConfig), Config)
    ->  true
    ;   member(auth(_, AuthConfig), Config)
    ->  true
    ;   AuthConfig = []  % No auth configured
    ).

app_auth_backend(AppSpec, Backend) :-
    app_auth_config(AppSpec, Config),
    (   member(backend(Backend), Config)
    ->  true
    ;   Backend = mock  % Default
    ).

app_auth_hash(AppSpec, Hash) :-
    app_auth_config(AppSpec, Config),
    (   member(password_hash(Hash), Config)
    ->  true
    ;   Hash = bcrypt  % Default
    ).

app_auth_token_type(AppSpec, TokenType) :-
    app_auth_config(AppSpec, Config),
    (   member(token_type(TokenType), Config)
    ->  true
    ;   TokenType = jwt  % Default
    ).

app_auth_users_file(AppSpec, UsersFile) :-
    app_auth_config(AppSpec, Config),
    (   member(users_file(UsersFile), Config)
    ->  true
    ;   UsersFile = 'users.txt'  % Default
    ).

app_auth_session_duration(AppSpec, Duration) :-
    app_auth_config(AppSpec, Config),
    (   member(session_duration(Duration), Config)
    ->  true
    ;   Duration = 86400  % Default: 24 hours
    ).

%% ============================================================================
%% Configuration Generation
%% ============================================================================

/**
 * generate_auth_config(+AppSpec, -ConfigJSON)
 *
 * Generates a JSON configuration file for the auth backend.
 */
generate_auth_config(AppSpec, ConfigJSON) :-
    app_auth_backend(AppSpec, Backend),
    app_auth_hash(AppSpec, Hash),
    app_auth_token_type(AppSpec, TokenType),
    app_auth_users_file(AppSpec, UsersFile),
    app_auth_session_duration(AppSpec, Duration),
    format(atom(ConfigJSON), '{
  "backend": "~w",
  "passwordHash": "~w",
  "tokenType": "~w",
  "usersFile": "~w",
  "sessionDuration": ~w
}', [Backend, Hash, TokenType, UsersFile, Duration]).

%% ============================================================================
%% Server Code Generation
%% ============================================================================

/**
 * generate_auth_server(+AppSpec, +Target, -Files)
 *
 * Generates auth server files for the specified target.
 */
generate_auth_server(AppSpec, node, Files) :-
    generate_auth_config(AppSpec, ConfigJSON),
    app_auth_backend(AppSpec, Backend),
    generate_node_auth_server(Backend, ServerCode),
    Files = [
        file('server/auth-config.json', ConfigJSON),
        file('server/auth-server.cjs', ServerCode)
    ].

generate_node_auth_server(text_file, Code) :-
    Code = '/**
 * Text File Authentication Server
 * Generated by UnifyWeaver
 */

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const CONFIG_PATH = path.join(__dirname, "auth-config.json");
const config = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));

const USERS_FILE = path.join(__dirname, config.usersFile);

// Ensure users file exists
if (!fs.existsSync(USERS_FILE)) {
  fs.writeFileSync(USERS_FILE, "# username:password_hash:roles:permissions\\n");
}

// Password hashing
function hashPassword(password) {
  if (config.passwordHash === "plain") {
    return password;
  } else if (config.passwordHash === "sha256") {
    const salt = crypto.randomBytes(16).toString("hex");
    const hash = crypto.createHash("sha256").update(password + salt).digest("hex");
    return `sha256:${salt}:${hash}`;
  } else if (config.passwordHash === "bcrypt") {
    // Requires bcrypt package
    const bcrypt = require("bcrypt");
    return bcrypt.hashSync(password, 12);
  }
  return password;
}

function verifyPassword(password, storedHash) {
  if (config.passwordHash === "plain") {
    return password === storedHash;
  } else if (config.passwordHash === "sha256") {
    const [algo, salt, hash] = storedHash.split(":");
    const testHash = crypto.createHash("sha256").update(password + salt).digest("hex");
    return hash === testHash;
  } else if (config.passwordHash === "bcrypt") {
    const bcrypt = require("bcrypt");
    return bcrypt.compareSync(password, storedHash);
  }
  return false;
}

// User operations
function loadUsers() {
  const content = fs.readFileSync(USERS_FILE, "utf-8");
  const users = {};
  for (const line of content.split("\\n")) {
    if (line.startsWith("#") || !line.trim()) continue;
    const [username, passwordHash, roles, permissions] = line.split(":");
    users[username] = {
      username,
      passwordHash,
      roles: roles ? roles.split(",") : [],
      permissions: permissions ? permissions.split(",") : []
    };
  }
  return users;
}

function saveUser(username, passwordHash, roles, permissions) {
  const line = `${username}:${passwordHash}:${roles.join(",")}:${permissions.join(",")}\\n`;
  fs.appendFileSync(USERS_FILE, line);
}

function authenticate(username, password) {
  const users = loadUsers();
  const user = users[username];
  if (!user) return null;
  if (!verifyPassword(password, user.passwordHash)) return null;
  return {
    username: user.username,
    roles: user.roles,
    permissions: user.permissions
  };
}

function register(username, password, roles = ["user"], permissions = ["read"]) {
  const users = loadUsers();
  if (users[username]) {
    throw new Error("User already exists");
  }
  const hash = hashPassword(password);
  saveUser(username, hash, roles, permissions);
  return { username, roles, permissions };
}

module.exports = { authenticate, register, loadUsers, hashPassword };
'.

generate_node_auth_server(mock, Code) :-
    Code = '/**
 * Mock Authentication Server
 * Generated by UnifyWeaver
 */

// Mock users - no persistence
const users = {
  "admin@test.com": { password: "admin123", roles: ["admin", "user"], permissions: ["read", "write", "delete"] },
  "shell@test.com": { password: "shell123", roles: ["shell", "admin", "user"], permissions: ["read", "write", "delete"] },
  "user@test.com": { password: "user123", roles: ["user"], permissions: ["read"] }
};

function authenticate(email, password) {
  const user = users[email.toLowerCase()];
  if (!user || user.password !== password) return null;
  return {
    email,
    roles: user.roles,
    permissions: user.permissions
  };
}

function register(email, password, roles = ["user"], permissions = ["read"]) {
  users[email.toLowerCase()] = { password, roles, permissions };
  return { email, roles, permissions };
}

module.exports = { authenticate, register };
'.

%% ============================================================================
%% Default Users Generation
%% ============================================================================

/**
 * generate_default_users(+AppSpec, -UsersContent)
 *
 * Generates default users file content.
 */
generate_default_users(AppSpec, Content) :-
    app_auth_hash(AppSpec, Hash),
    (   Hash = plain
    ->  Content = '# UnifyWeaver Users File
# Format: username:password:roles:permissions
admin:admin123:admin,user:read,write,delete
shell:shell123:shell,admin,user:read,write,delete
user:user123:user:read
guest:guest123:guest:read
'
    ;   Content = '# UnifyWeaver Users File
# Format: username:password_hash:roles:permissions
# Passwords must be hashed before adding
# Use: node -e "require(\'./auth-server.cjs\').hashPassword(\'password\')"
'
    ).
