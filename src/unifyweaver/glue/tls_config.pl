/**
 * tls_config.pl - Declarative TLS/SSL Configuration
 *
 * This module provides declarative specification of TLS/SSL configurations
 * for UnifyWeaver-generated applications.
 *
 * Usage:
 *   :- use_module(tls_config).
 *
 *   % Define app with TLS configuration
 *   app(my_app, [
 *       tls([
 *           mode(proxy),                    % TLS termination strategy
 *           cert_source(auto_generated),    % Certificate source
 *           https_port(443),                % HTTPS port
 *           ...
 *       ])
 *   ]).
 *
 * TLS Modes:
 *   - proxy (default)    : Dev proxy terminates TLS, backends use plain HTTP
 *   - shared_cert        : All servers share the same certificate file
 *   - reverse_proxy      : External reverse proxy (nginx/caddy) handles TLS
 *   - per_server         : Each server manages its own certificate
 *
 * Certificate Sources:
 *   - auto_generated     : Generate self-signed cert (development)
 *   - lets_encrypt       : Use Let's Encrypt ACME (production)
 *   - file(CertPath,Key) : Load from files
 *   - env(CertVar,KeyVar): Load from environment variables
 *
 * @author UnifyWeaver
 * @version 1.0.0
 */

:- module(tls_config, [
    % TLS mode definitions
    tls_mode/2,
    mode_available/1,
    mode_description/2,
    mode_requirements/2,

    % Certificate source definitions
    cert_source/2,
    cert_source_available/1,

    % App TLS extraction
    app_tls_config/2,
    app_tls_mode/2,
    app_cert_source/2,
    app_https_port/2,

    % Code generation
    generate_tls_config/2,
    generate_vite_proxy_config/2,
    generate_nginx_config/3,
    generate_caddy_config/3,
    generate_shared_cert_loader/2
]).

:- use_module(library(lists)).

%% ============================================================================
%% TLS Mode Definitions
%% ============================================================================

/**
 * tls_mode(+Mode, +Properties)
 *
 * Defines available TLS termination modes.
 */

tls_mode(proxy, [
    description('Development proxy terminates TLS, backends use plain HTTP'),
    use_case(development),
    advantages([
        'Single certificate to manage',
        'Easy development setup',
        'Hot reload works seamlessly',
        'No backend TLS configuration needed'
    ]),
    disadvantages([
        'Requires proxy server running',
        'Internal traffic unencrypted (localhost only)',
        'Not suitable for distributed backends'
    ]),
    requirements([
        vite_or_proxy_server
    ]),
    status(implemented),
    default(true)
]).

tls_mode(shared_cert, [
    description('All servers share the same certificate files'),
    use_case(production),
    advantages([
        'End-to-end encryption',
        'Servers can run independently',
        'Direct access to any server'
    ]),
    disadvantages([
        'Certificate files must be distributed',
        'Each server needs TLS configuration',
        'Multiple ports exposed'
    ]),
    requirements([
        cert_file,
        key_file
    ]),
    status(implemented)
]).

tls_mode(reverse_proxy, [
    description('External reverse proxy handles TLS termination'),
    use_case(production),
    advantages([
        'Professional grade TLS termination',
        'Load balancing support',
        'Central certificate management',
        'HTTP/2 and HTTP/3 support'
    ]),
    disadvantages([
        'Additional infrastructure required',
        'More complex setup'
    ]),
    proxy_options([nginx, caddy, haproxy, traefik]),
    requirements([
        external_proxy
    ]),
    status(implemented)
]).

tls_mode(per_server, [
    description('Each server generates and manages its own certificate'),
    use_case(development),
    advantages([
        'No shared state',
        'Servers fully independent',
        'Easy to add new servers'
    ]),
    disadvantages([
        'Multiple certificates to accept',
        'Certificate management overhead',
        'Inconsistent security posture'
    ]),
    requirements([]),
    status(implemented)
]).

mode_available(Mode) :-
    tls_mode(Mode, Props),
    member(status(implemented), Props).

mode_description(Mode, Desc) :-
    tls_mode(Mode, Props),
    member(description(Desc), Props).

mode_requirements(Mode, Reqs) :-
    tls_mode(Mode, Props),
    member(requirements(Reqs), Props).

%% ============================================================================
%% Certificate Source Definitions
%% ============================================================================

cert_source(auto_generated, [
    description('Auto-generate self-signed certificate'),
    use_case(development),
    validity_days(365),
    requirements([openssl]),
    status(implemented)
]).

cert_source(lets_encrypt, [
    description('Obtain certificate from Let\'s Encrypt'),
    use_case(production),
    requirements([
        domain_name,
        public_access,
        acme_client
    ]),
    acme_clients([certbot, acme_sh, lego]),
    status(proposed)
]).

cert_source(file, [
    description('Load certificate from files'),
    use_case(any),
    parameters([cert_path, key_path]),
    requirements([cert_file, key_file]),
    status(implemented)
]).

cert_source(env, [
    description('Load certificate from environment variables'),
    use_case(container),
    parameters([cert_var, key_var]),
    requirements([env_vars]),
    status(implemented)
]).

cert_source(vault, [
    description('Load certificate from HashiCorp Vault'),
    use_case(enterprise),
    requirements([vault_server, vault_token]),
    status(proposed)
]).

cert_source_available(Source) :-
    cert_source(Source, Props),
    member(status(implemented), Props).

%% ============================================================================
%% App TLS Extraction
%% ============================================================================

/**
 * app_tls_config(+AppSpec, -TlsConfig)
 *
 * Extracts TLS configuration from an app specification.
 */
app_tls_config(app(_, Config), TlsConfig) :-
    (   member(tls(TlsConfig), Config)
    ->  true
    ;   TlsConfig = []  % No TLS configured, use defaults
    ).

app_tls_mode(AppSpec, Mode) :-
    app_tls_config(AppSpec, Config),
    (   member(mode(Mode), Config)
    ->  true
    ;   Mode = proxy  % Default
    ).

app_cert_source(AppSpec, Source) :-
    app_tls_config(AppSpec, Config),
    (   member(cert_source(Source), Config)
    ->  true
    ;   Source = auto_generated  % Default
    ).

app_https_port(AppSpec, Port) :-
    app_tls_config(AppSpec, Config),
    (   member(https_port(Port), Config)
    ->  true
    ;   Port = 443  % Default
    ).

app_cert_path(AppSpec, CertPath) :-
    app_tls_config(AppSpec, Config),
    (   member(cert_path(CertPath), Config)
    ->  true
    ;   CertPath = 'certs/server.crt'  % Default
    ).

app_key_path(AppSpec, KeyPath) :-
    app_tls_config(AppSpec, Config),
    (   member(key_path(KeyPath), Config)
    ->  true
    ;   KeyPath = 'certs/server.key'  % Default
    ).

%% ============================================================================
%% Configuration Generation
%% ============================================================================

/**
 * generate_tls_config(+AppSpec, -ConfigJSON)
 *
 * Generates a JSON TLS configuration file.
 */
generate_tls_config(AppSpec, ConfigJSON) :-
    app_tls_mode(AppSpec, Mode),
    app_cert_source(AppSpec, CertSource),
    app_https_port(AppSpec, HttpsPort),
    app_cert_path(AppSpec, CertPath),
    app_key_path(AppSpec, KeyPath),
    format(atom(ConfigJSON), '{
  "mode": "~w",
  "certSource": "~w",
  "httpsPort": ~w,
  "certPath": "~w",
  "keyPath": "~w"
}', [Mode, CertSource, HttpsPort, CertPath, KeyPath]).

%% ============================================================================
%% Vite Proxy Configuration (mode: proxy)
%% ============================================================================

/**
 * generate_vite_proxy_config(+Backends, -ViteConfig)
 *
 * Generates Vite proxy configuration for backend services.
 * Backends is a list of backend(Name, Port, PathPrefix) terms.
 */
generate_vite_proxy_config(Backends, ViteConfig) :-
    maplist(format_vite_proxy_entry, Backends, Entries),
    atomic_list_concat(Entries, ',\n      ', EntriesStr),
    format(atom(ViteConfig), '// Vite proxy configuration (TLS mode: proxy)
// All backends accessible through single HTTPS port
export default {
  server: {
    https: true,
    proxy: {
      ~w
    }
  }
}', [EntriesStr]).

format_vite_proxy_entry(backend(Name, Port, PathPrefix, websocket), Entry) :-
    !,
    format(atom(Entry), '\'~w\': {
        target: \'ws://localhost:~w\',
        ws: true,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\\~w/, \'\')
      }', [PathPrefix, Port, PathPrefix]).

format_vite_proxy_entry(backend(Name, Port, PathPrefix), Entry) :-
    format(atom(Entry), '\'~w\': {
        target: \'http://localhost:~w\',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\\~w/, \'\')
      }', [PathPrefix, Port, PathPrefix]).

%% ============================================================================
%% Nginx Configuration (mode: reverse_proxy)
%% ============================================================================

/**
 * generate_nginx_config(+AppSpec, +Backends, -NginxConfig)
 *
 * Generates nginx reverse proxy configuration.
 */
generate_nginx_config(AppSpec, Backends, NginxConfig) :-
    app_https_port(AppSpec, HttpsPort),
    app_cert_path(AppSpec, CertPath),
    app_key_path(AppSpec, KeyPath),
    maplist(format_nginx_upstream, Backends, Upstreams),
    maplist(format_nginx_location, Backends, Locations),
    atomic_list_concat(Upstreams, '\n', UpstreamsStr),
    atomic_list_concat(Locations, '\n\n    ', LocationsStr),
    format(atom(NginxConfig), '# Nginx reverse proxy configuration
# Generated by UnifyWeaver (TLS mode: reverse_proxy)

~w

server {
    listen ~w ssl http2;
    server_name localhost;

    ssl_certificate ~w;
    ssl_certificate_key ~w;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend (static files or dev server)
    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    ~w
}
', [UpstreamsStr, HttpsPort, CertPath, KeyPath, LocationsStr]).

format_nginx_upstream(backend(Name, Port, _), Upstream) :-
    format(atom(Upstream), 'upstream ~w_backend {
    server localhost:~w;
}', [Name, Port]).

format_nginx_upstream(backend(Name, Port, _, websocket), Upstream) :-
    format(atom(Upstream), 'upstream ~w_backend {
    server localhost:~w;
}', [Name, Port]).

format_nginx_location(backend(Name, Port, PathPrefix), Location) :-
    format(atom(Location), '# ~w API
    location ~w {
        proxy_pass http://~w_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }', [Name, PathPrefix, Name]).

format_nginx_location(backend(Name, Port, PathPrefix, websocket), Location) :-
    format(atom(Location), '# ~w WebSocket
    location ~w {
        proxy_pass http://~w_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }', [Name, PathPrefix, Name]).

%% ============================================================================
%% Caddy Configuration (mode: reverse_proxy)
%% ============================================================================

/**
 * generate_caddy_config(+AppSpec, +Backends, -CaddyConfig)
 *
 * Generates Caddy reverse proxy configuration.
 */
generate_caddy_config(AppSpec, Backends, CaddyConfig) :-
    app_https_port(AppSpec, HttpsPort),
    app_cert_path(AppSpec, CertPath),
    app_key_path(AppSpec, KeyPath),
    maplist(format_caddy_route, Backends, Routes),
    atomic_list_concat(Routes, '\n\n    ', RoutesStr),
    format(atom(CaddyConfig), '# Caddy reverse proxy configuration
# Generated by UnifyWeaver (TLS mode: reverse_proxy)

localhost:~w {
    tls ~w ~w

    ~w

    # Frontend fallback
    reverse_proxy localhost:5173
}
', [HttpsPort, CertPath, KeyPath, RoutesStr]).

format_caddy_route(backend(Name, Port, PathPrefix), Route) :-
    format(atom(Route), '# ~w API
    handle_path ~w* {
        reverse_proxy localhost:~w
    }', [Name, PathPrefix, Port]).

format_caddy_route(backend(Name, Port, PathPrefix, websocket), Route) :-
    format(atom(Route), '# ~w WebSocket
    @~w_ws {
        path ~w*
        header Connection *Upgrade*
        header Upgrade websocket
    }
    reverse_proxy @~w_ws localhost:~w', [Name, Name, PathPrefix, Name, Port]).

%% ============================================================================
%% Shared Certificate Loader (mode: shared_cert)
%% ============================================================================

/**
 * generate_shared_cert_loader(+AppSpec, -LoaderCode)
 *
 * Generates Node.js code to load shared certificates.
 */
generate_shared_cert_loader(AppSpec, LoaderCode) :-
    app_cert_path(AppSpec, CertPath),
    app_key_path(AppSpec, KeyPath),
    format(atom(LoaderCode), '/**
 * Shared TLS Certificate Loader
 * Generated by UnifyWeaver (TLS mode: shared_cert)
 *
 * Usage:
 *   const { createSecureServer } = require(\'./tls-loader.cjs\');
 *   const server = createSecureServer(app);
 *   server.listen(443);
 */

const fs = require(\'fs\');
const path = require(\'path\');
const https = require(\'https\');

const CERT_PATH = process.env.TLS_CERT_PATH || \'~w\';
const KEY_PATH = process.env.TLS_KEY_PATH || \'~w\';

let tlsOptions = null;

function loadCertificates() {
  const certPath = path.resolve(__dirname, CERT_PATH);
  const keyPath = path.resolve(__dirname, KEY_PATH);

  if (!fs.existsSync(certPath) || !fs.existsSync(keyPath)) {
    console.warn(\'TLS certificates not found, generating self-signed...\');
    generateSelfSigned(certPath, keyPath);
  }

  tlsOptions = {
    cert: fs.readFileSync(certPath),
    key: fs.readFileSync(keyPath)
  };

  console.log(\'TLS certificates loaded:\');
  console.log(\'  Cert:\', certPath);
  console.log(\'  Key:\', keyPath);

  return tlsOptions;
}

function generateSelfSigned(certPath, keyPath) {
  const { execSync } = require(\'child_process\');
  const certDir = path.dirname(certPath);

  if (!fs.existsSync(certDir)) {
    fs.mkdirSync(certDir, { recursive: true });
  }

  execSync(`openssl req -x509 -newkey rsa:4096 -keyout \"${keyPath}\" -out \"${certPath}\" -days 365 -nodes -subj \"/CN=localhost\"`, {
    stdio: \'pipe\'
  });

  console.log(\'Generated self-signed certificate\');
}

function getTlsOptions() {
  if (!tlsOptions) {
    loadCertificates();
  }
  return tlsOptions;
}

function createSecureServer(requestHandler) {
  return https.createServer(getTlsOptions(), requestHandler);
}

// WebSocket server with TLS
function createSecureWebSocketServer(options = {}) {
  const { WebSocketServer } = require(\'ws\');
  const server = https.createServer(getTlsOptions());
  const wss = new WebSocketServer({ server, ...options });
  return { server, wss };
}

module.exports = {
  loadCertificates,
  getTlsOptions,
  createSecureServer,
  createSecureWebSocketServer,
  CERT_PATH,
  KEY_PATH
};
', [CertPath, KeyPath]).

%% ============================================================================
%% Default Configuration
%% ============================================================================

/**
 * default_tls_config(-Config)
 *
 * Returns the default TLS configuration.
 */
default_tls_config([
    mode(proxy),
    cert_source(auto_generated),
    https_port(443),
    cert_path('certs/server.crt'),
    key_path('certs/server.key')
]).
