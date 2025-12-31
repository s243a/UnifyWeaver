% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Full Pipeline Generator - Complete Full-Stack Application Generation
%
% This module ties together all UnifyWeaver glue modules to generate
% complete full-stack applications from declarative Prolog specifications.
%
% Usage:
%   % Define an application
%   application(python_bridge_demo, [
%       backend([
%           server(express, port(3001)),
%           ffi_bridge(rust, lib(librpyc_bridge)),
%           rpyc_server(localhost, 18812)
%       ]),
%       api([
%           include_endpoints(numpy_endpoints),
%           include_endpoints(math_endpoints),
%           security(rpyc_security_rules)
%       ]),
%       frontend([
%           framework(react),
%           components([numpy_calculator, math_calculator]),
%           theme(default)
%       ])
%   ]).
%
%   % Generate the application
%   ?- generate_application(python_bridge_demo, OutputDir).

:- module(full_pipeline, [
    % Application specification
    application/2,                      % application(+Name, +Config)

    % Application management
    declare_application/2,              % declare_application(+Name, +Config)
    clear_applications/0,               % clear_applications
    all_applications/1,                 % all_applications(-Apps)

    % Full generation
    generate_application/2,             % generate_application(+Name, -Files)
    generate_application/3,             % generate_application(+Name, +Options, -Files)

    % Individual file generators
    generate_package_json/3,            % generate_package_json(+Name, +Config, -Code)
    generate_tsconfig/2,                % generate_tsconfig(+Config, -Code)
    generate_dockerfile/3,              % generate_dockerfile(+Name, +Config, -Code)
    generate_readme/3,                  % generate_readme(+Name, +Config, -Code)

    % Output utilities
    write_application_files/2,          % write_application_files(+OutputDir, +Files)

    % Testing
    test_full_pipeline/0
]).

:- use_module(library(lists)).

% Import other glue modules
:- use_module('./rpyc_security').
:- use_module('./express_generator').
:- use_module('./react_generator').

% Import preferences/firewall config (optional - graceful if not found)
:- catch(use_module('./typescript_glue_config'), _, true).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic application/2.

:- discontiguous application/2.

% ============================================================================
% DEFAULT APPLICATION SPECIFICATIONS
% ============================================================================

% Demo application showcasing all features
application(python_bridge_demo, [
    name("Python Bridge Demo"),
    description("Full-stack demo of Python ↔ TypeScript via RPyC"),
    version("0.1.0"),

    backend([
        server(express, [port(3001)]),
        ffi_bridge(rust, [lib_name(rpyc_bridge)]),
        rpyc_server([host(localhost), port(18812)])
    ]),

    api([
        include_endpoints([math_endpoints, numpy_endpoints, statistics_endpoints]),
        security([
            whitelist(true),
            rate_limit(true),
            validation(true)
        ])
    ]),

    frontend([
        framework(react),
        components([numpy_calculator, math_calculator, statistics_calculator, math_constants]),
        theme(default),
        port(3000)
    ]),

    deployment([
        docker(true),
        compose(true)
    ])
]).

% Minimal API-only application
application(api_only, [
    name("API Server"),
    description("Minimal RPyC API server"),
    version("0.1.0"),

    backend([
        server(express, [port(3001)])
    ]),

    api([
        include_endpoints([math_endpoints]),
        security([whitelist(true)])
    ])
]).

% ============================================================================
% APPLICATION MANAGEMENT
% ============================================================================

%% declare_application(+Name, +Config)
%  Dynamically declare an application.
declare_application(Name, Config) :-
    (   application(Name, _)
    ->  retract(application(Name, _))
    ;   true
    ),
    assertz(application(Name, Config)).

%% clear_applications
%  Clear all dynamic applications.
clear_applications :-
    retractall(application(_, _)).

%% all_applications(-Apps)
%  Get all defined applications.
all_applications(Apps) :-
    findall(Name-Config, application(Name, Config), Apps).

% ============================================================================
% FULL APPLICATION GENERATION
% ============================================================================

%% generate_application(+Name, -Files)
%  Generate all files for an application.
generate_application(Name, Files) :-
    generate_application(Name, [], Files).

%% generate_application(+Name, +Options, -Files)
%  Generate all files for an application with options.
%  Options can include context(production|development|testing) to use preferences.
generate_application(Name, Options, Files) :-
    application(Name, Config),
    % Set generation context from options if available
    (   member(context(Context), Options),
        current_module(typescript_glue_config)
    ->  typescript_glue_config:set_generation_context(Context)
    ;   true
    ),
    % Merge config with preferences-based defaults
    get_merged_config(Config, Options, MergedConfig),
    % Collect all generated files
    findall(File, generate_app_file(Name, MergedConfig, File), Files),
    % Clean up context
    (   current_module(typescript_glue_config)
    ->  catch(retractall(typescript_glue_config:current_generation_context(_)), _, true)
    ;   true
    ).

%% get_merged_config(+BaseConfig, +Options, -MergedConfig) is det.
%
% Merge application config with preferences from typescript_glue_config.
get_merged_config(BaseConfig, Options, MergedConfig) :-
    (   current_module(typescript_glue_config)
    ->  % Get pipeline config from preferences
        catch(typescript_glue_config:get_pipeline_config(Options, PipelineConfig), _, PipelineConfig = []),
        % Merge pipeline config into base config
        merge_app_config(BaseConfig, PipelineConfig, MergedConfig)
    ;   MergedConfig = BaseConfig
    ).

%% merge_app_config(+Base, +Override, -Merged) is det.
%
% Merge application configurations.
merge_app_config(Base, Override, Merged) :-
    findall(Term, (
        member(Term, Override)
    ;   member(Term, Base),
        functor(Term, Key, _),
        \+ (member(OTerm, Override), functor(OTerm, Key, _))
    ), Merged).

%% generate_app_file(+Name, +Config, -File)
%  Generate a single file for the application.
%  File = file(Path, Content)

% Package.json
generate_app_file(Name, Config, file('package.json', Content)) :-
    generate_package_json(Name, Config, Content).

% tsconfig.json
generate_app_file(_Name, Config, file('tsconfig.json', Content)) :-
    generate_tsconfig(Config, Content).

% Backend files
generate_app_file(Name, Config, file('src/server/index.ts', Content)) :-
    member(backend(_), Config),
    generate_express_app(Name, Content).

generate_app_file(Name, Config, file('src/server/router.ts', Content)) :-
    member(api(ApiConfig), Config),
    (member(include_endpoints(Groups), ApiConfig) -> true ; Groups = []),
    generate_express_router(Name, [endpoints(Groups)], Content).

generate_app_file(_Name, Config, file('src/server/whitelist.ts', Content)) :-
    member(api(ApiConfig), Config),
    member(security(SecConfig), ApiConfig),
    member(whitelist(true), SecConfig),
    generate_typescript_whitelist(Content).

generate_app_file(_Name, Config, file('src/server/validator.ts', Content)) :-
    member(api(ApiConfig), Config),
    member(security(SecConfig), ApiConfig),
    member(validation(true), SecConfig),
    generate_typescript_validator(Content).

generate_app_file(_Name, Config, file('src/server/security_middleware.ts', Content)) :-
    member(api(ApiConfig), Config),
    member(security(SecConfig), ApiConfig),
    member(rate_limit(true), SecConfig),
    generate_express_security_middleware(Content).

% Frontend files
generate_app_file(_Name, Config, file(Path, Content)) :-
    member(frontend(FEConfig), Config),
    member(components(CompNames), FEConfig),
    member(CompName, CompNames),
    ui_component(CompName, _),
    atom_string(CompName, CompNameStr),
    pascal_case(CompNameStr, PascalName),
    format(atom(Path), 'src/components/~w/~w.tsx', [PascalName, PascalName]),
    generate_react_component(CompName, Content).

generate_app_file(_Name, Config, file(Path, Content)) :-
    member(frontend(FEConfig), Config),
    member(components(CompNames), FEConfig),
    member(CompName, CompNames),
    ui_component(CompName, _),
    atom_string(CompName, CompNameStr),
    pascal_case(CompNameStr, PascalName),
    format(atom(Path), 'src/components/~w/~w.module.css', [PascalName, PascalName]),
    generate_component_styles(CompName, Content).

generate_app_file(Name, Config, file('src/App.tsx', Content)) :-
    member(frontend(FEConfig), Config),
    member(components(CompNames), FEConfig),
    generate_react_app(Name, [components(CompNames)], Content).

generate_app_file(_Name, Config, file('src/hooks/useApi.ts', Content)) :-
    member(frontend(_), Config),
    generate_api_hooks(python, Content).

% Docker files
generate_app_file(Name, Config, file('Dockerfile', Content)) :-
    member(deployment(DepConfig), Config),
    member(docker(true), DepConfig),
    generate_dockerfile(Name, Config, Content).

generate_app_file(Name, Config, file('docker-compose.yml', Content)) :-
    member(deployment(DepConfig), Config),
    member(compose(true), DepConfig),
    generate_docker_compose(Name, Config, Content).

% README
generate_app_file(Name, Config, file('README.md', Content)) :-
    generate_readme(Name, Config, Content).

% ============================================================================
% PACKAGE.JSON GENERATION
% ============================================================================

%% generate_package_json(+Name, +Config, -Code)
%  Generate package.json for the application.
generate_package_json(Name, Config, Code) :-
    atom_string(Name, NameStr),

    (member(description(Desc), Config) -> true ; Desc = "Generated by UnifyWeaver"),
    (member(version(Version), Config) -> true ; Version = "0.1.0"),

    % Determine if frontend is included
    (member(frontend(_), Config) -> HasFrontend = true ; HasFrontend = false),

    % Generate scripts based on configuration
    (   HasFrontend == true
    ->  Scripts = '"dev": "concurrently \\"npm run dev:server\\" \\"npm run dev:client\\"",
    "dev:server": "ts-node-dev src/server/index.ts",
    "dev:client": "vite",
    "build": "tsc && vite build",
    "start": "node dist/server/index.js"'
    ;   Scripts = '"dev": "ts-node-dev src/server/index.ts",
    "build": "tsc",
    "start": "node dist/server/index.js"'
    ),

    % Generate dependencies
    (   HasFrontend == true
    ->  FrontendDeps = ',
    "react": "^18.2.0",
    "react-dom": "^18.2.0"'
    ;   FrontendDeps = ''
    ),

    (   HasFrontend == true
    ->  FrontendDevDeps = ',
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.0.0",
    "concurrently": "^8.2.0"'
    ;   FrontendDevDeps = ''
    ),

    format(atom(Code), '{
  "name": "~w",
  "version": "~w",
  "description": "~w",
  "type": "module",
  "scripts": {
    ~w
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "koffi": "^2.8.0"~w
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/cors": "^2.8.17",
    "@types/node": "^20.10.0",
    "typescript": "^5.3.0",
    "ts-node-dev": "^2.0.0"~w
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
', [NameStr, Version, Desc, Scripts, FrontendDeps, FrontendDevDeps]).

% ============================================================================
% TSCONFIG GENERATION
% ============================================================================

%% generate_tsconfig(+Config, -Code)
%  Generate tsconfig.json.
generate_tsconfig(Config, Code) :-
    (member(frontend(_), Config) -> JSX = '"jsx": "react-jsx",' ; JSX = ''),

    format(atom(Code), '{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "node",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "outDir": "./dist",
    "rootDir": "./src",
    ~w
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
', [JSX]).

% ============================================================================
% DOCKERFILE GENERATION
% ============================================================================

%% generate_dockerfile(+Name, +Config, -Code)
%  Generate Dockerfile for the application.
generate_dockerfile(_Name, Config, Code) :-
    % Get backend port
    (   member(backend(BEConfig), Config),
        member(server(_, ServerOpts), BEConfig),
        member(port(Port), ServerOpts)
    ->  true
    ;   Port = 3001
    ),

    format(atom(Code), '# Generated by UnifyWeaver
FROM node:20-slim

# Install Python for RPyC
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install rpyc numpy

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy built application
COPY dist ./dist

# Expose port
EXPOSE ~w

# Start the application
CMD ["node", "dist/server/index.js"]
', [Port]).

%% generate_docker_compose(+Name, +Config, -Code)
%  Generate docker-compose.yml.
generate_docker_compose(Name, Config, Code) :-
    atom_string(Name, NameStr),

    % Get ports
    (   member(backend(BEConfig), Config),
        member(server(_, ServerOpts), BEConfig),
        member(port(BackendPort), ServerOpts)
    ->  true
    ;   BackendPort = 3001
    ),

    (   member(frontend(FEConfig), Config),
        member(port(_FrontendPort), FEConfig)
    ->  true
    ;   true  % FrontendPort reserved for future frontend container
    ),

    % Get RPyC config
    (   member(backend(BEConfig2), Config),
        member(rpyc_server(RPyCOpts), BEConfig2),
        member(port(RPyCPort), RPyCOpts)
    ->  true
    ;   RPyCPort = 18812
    ),

    format(atom(Code), '# Generated by UnifyWeaver
version: "3.8"

services:
  # Python RPyC server
  rpyc-server:
    build:
      context: .
      dockerfile: Dockerfile.rpyc
    ports:
      - "~w:~w"
    volumes:
      - ./python:/app
    networks:
      - ~w-network

  # Node.js API server
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "~w:~w"
    environment:
      - RPYC_HOST=rpyc-server
      - RPYC_PORT=~w
    depends_on:
      - rpyc-server
    networks:
      - ~w-network

networks:
  ~w-network:
    driver: bridge
', [RPyCPort, RPyCPort, NameStr, BackendPort, BackendPort, RPyCPort, NameStr, NameStr]).

% ============================================================================
% README GENERATION
% ============================================================================

%% generate_readme(+Name, +Config, -Code)
%  Generate README.md for the application.
generate_readme(Name, Config, Code) :-
    atom_string(Name, NameStr),

    (member(name(DisplayName), Config) -> true ; DisplayName = NameStr),
    (member(description(Desc), Config) -> true ; Desc = "Generated by UnifyWeaver"),

    % Determine features
    (member(frontend(_), Config) -> HasFrontend = "Yes" ; HasFrontend = "No"),
    (member(api(ApiConfig), Config), member(security(_), ApiConfig)
    ->  HasSecurity = "Yes" ; HasSecurity = "No"),
    (member(deployment(DepConfig), Config), member(docker(true), DepConfig)
    ->  HasDocker = "Yes" ; HasDocker = "No"),

    format(atom(Code), '# ~w

~w

> Generated by [UnifyWeaver](https://github.com/s243a/UnifyWeaver)

## Features

| Feature | Status |
|---------|--------|
| Backend (Express) | Yes |
| Frontend (React) | ~w |
| Security (Whitelist) | ~w |
| Docker | ~w |

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.9+ with `rpyc` and `numpy`
- Rust toolchain (for FFI bridge)

### Installation

```bash
# Install dependencies
npm install

# Start RPyC server (in separate terminal)
python3 -c "import rpyc; rpyc.utils.server.ThreadedServer(rpyc.SlaveService).start()"

# Build Rust FFI bridge
cd rust-bridge && cargo build --release && cd ..

# Start development server
npm run dev
```

### Docker

```bash
docker-compose up --build
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React App     │────▶│  Express API     │────▶│  Rust FFI       │
│   (Frontend)    │     │  (Backend)       │     │  Bridge         │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Python/RPyC    │
                                                 │  (NumPy, etc.)  │
                                                 └─────────────────┘
```

## API Endpoints

The API provides secure access to Python functions via RPyC:

- `POST /api/numpy/mean` - Calculate mean
- `POST /api/numpy/std` - Calculate standard deviation
- `POST /api/math/sqrt` - Calculate square root
- `GET /api/math/pi` - Get pi constant

See `src/server/router.ts` for the full list.

## Security

All API calls are validated against a whitelist of allowed Python modules
and functions. Rate limiting and input sanitization are enabled by default.

## License

MIT OR Apache-2.0
', [DisplayName, Desc, HasFrontend, HasSecurity, HasDocker]).

% ============================================================================
% FILE OUTPUT
% ============================================================================

%% write_application_files(+OutputDir, +Files)
%  Write all generated files to the output directory.
write_application_files(OutputDir, Files) :-
    forall(member(file(RelPath, Content), Files), (
        atomic_list_concat([OutputDir, '/', RelPath], FullPath),
        % Ensure directory exists
        file_directory_name(FullPath, Dir),
        (exists_directory(Dir) -> true ; make_directory_path(Dir)),
        % Write file
        open(FullPath, write, Stream),
        write(Stream, Content),
        close(Stream),
        format('  Created: ~w~n', [RelPath])
    )).

% ============================================================================
% HELPER: pascal_case from react_generator
% ============================================================================

pascal_case(Input, Output) :-
    atom_string(InputAtom, Input),
    atomic_list_concat(Parts, '_', InputAtom),
    maplist(capitalize_first_atom, Parts, CapParts),
    atomic_list_concat(CapParts, '', Output).

capitalize_first_atom(Atom, Capitalized) :-
    atom_string(Atom, Str),
    (   Str = ""
    ->  Capitalized = ''
    ;   string_codes(Str, [First|Rest]),
        to_upper(First, Upper),
        string_codes(CapStr, [Upper|Rest]),
        atom_string(Capitalized, CapStr)
    ).

% ============================================================================
% TESTING
% ============================================================================

test_full_pipeline :-
    format('~n=== Full Pipeline Tests ===~n~n'),

    % Test application queries
    format('Application Queries:~n'),
    all_applications(AllApps),
    length(AllApps, AppCount),
    format('  Total applications: ~w~n', [AppCount]),

    % Test file generation
    format('~nFile Generation:~n'),
    (   generate_application(python_bridge_demo, Files),
        length(Files, FileCount),
        format('  python_bridge_demo: ~w files~n', [FileCount]),
        forall(member(file(Path, Content), Files), (
            atom_length(Content, Len),
            format('    ~w: ~d chars~n', [Path, Len])
        ))
    ;   format('  python_bridge_demo: FAILED~n')
    ),

    format('~n=== Tests Complete ===~n').
