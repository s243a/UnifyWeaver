# Makefile for Pyodide Matrix Example

.PHONY: all generate serve clean

all: generate

# Generate Python module and TypeScript bindings from Prolog
generate:
	swipl -g "matrix_module:generate_all" -t halt matrix_module.pl

# Start a local development server
serve:
	@echo "Starting server at http://localhost:8080"
	@echo "Press Ctrl+C to stop"
	python3 -m http.server 8080

# Alternative: use Node.js server
serve-node:
	@echo "Starting server at http://localhost:8080"
	npx http-server -p 8080

# Clean generated files
clean:
	rm -f matrix_lib.py matrix_wasm.ts

# Open in browser (Linux)
open:
	xdg-open index.html 2>/dev/null || firefox index.html

# Open in browser (macOS)
open-mac:
	open index.html
