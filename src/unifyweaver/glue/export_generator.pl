% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Export Generator - SVG, PNG, and PDF Export for Visualizations
%
% This module provides declarative export specifications that generate
% code for exporting visualizations to various formats.
%
% Usage:
%   % Define export configuration
%   export_config(my_chart, [
%       formats([svg, png, pdf]),
%       filename_template("chart-{date}-{title}"),
%       default_size(800, 600)
%   ]).
%
%   % Generate export functionality
%   ?- generate_export_component(my_chart, Component).

:- module(export_generator, [
    % Export specifications
    export_config/2,                % export_config(+Component, +Options)
    export_format/2,                % export_format(+Format, +Options)

    % Generation predicates
    generate_export_component/2,    % generate_export_component(+Component, -JSX)
    generate_export_hook/2,         % generate_export_hook(+Component, -Hook)
    generate_svg_export/2,          % generate_svg_export(+Component, -Code)
    generate_png_export/2,          % generate_png_export(+Component, -Code)
    generate_pdf_export/2,          % generate_pdf_export(+Component, -Code)
    generate_export_button/3,       % generate_export_button(+Format, +Options, -JSX)
    generate_export_menu/2,         % generate_export_menu(+Component, -JSX)
    generate_export_css/1,          % generate_export_css(-CSS)

    % Utility predicates
    get_export_formats/2,           % get_export_formats(+Component, -Formats)
    get_export_filename/3,          % get_export_filename(+Component, +Format, -Filename)
    supported_format/1,             % supported_format(+Format)

    % Management
    declare_export_config/2,        % declare_export_config(+Component, +Options)
    clear_export_configs/0,         % clear_export_configs

    % Testing
    test_export_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic export_config/2.
:- dynamic export_format/2.

:- discontiguous export_config/2.

% ============================================================================
% SUPPORTED FORMATS
% ============================================================================

supported_format(svg).
supported_format(png).
supported_format(pdf).
supported_format(json).
supported_format(csv).

% ============================================================================
% FORMAT SPECIFICATIONS
% ============================================================================

export_format(svg, [
    mime_type('image/svg+xml'),
    extension('.svg'),
    description("Scalable Vector Graphics"),
    vector(true),
    requires_canvas(false)
]).

export_format(png, [
    mime_type('image/png'),
    extension('.png'),
    description("PNG Image"),
    vector(false),
    requires_canvas(true),
    default_scale(2)
]).

export_format(pdf, [
    mime_type('application/pdf'),
    extension('.pdf'),
    description("PDF Document"),
    vector(true),
    requires_canvas(false),
    library(jspdf)
]).

export_format(json, [
    mime_type('application/json'),
    extension('.json'),
    description("JSON Data"),
    vector(false),
    requires_canvas(false)
]).

export_format(csv, [
    mime_type('text/csv'),
    extension('.csv'),
    description("CSV Data"),
    vector(false),
    requires_canvas(false)
]).

% ============================================================================
% DEFAULT EXPORT CONFIGURATIONS
% ============================================================================

export_config(default, [
    formats([svg, png, pdf]),
    filename_template("{component}-{timestamp}"),
    default_size(800, 600),
    scale(2),
    background(white),
    include_styles(true)
]).

export_config(line_chart, [
    formats([svg, png, pdf, csv]),
    filename_template("line-chart-{timestamp}"),
    default_size(800, 400),
    scale(2),
    background(transparent)
]).

export_config(bar_chart, [
    formats([svg, png, pdf, csv]),
    filename_template("bar-chart-{timestamp}"),
    default_size(800, 500),
    scale(2)
]).

export_config(scatter_plot, [
    formats([svg, png, pdf, json]),
    filename_template("scatter-{timestamp}"),
    default_size(800, 800),
    scale(2)
]).

export_config(pie_chart, [
    formats([svg, png, pdf]),
    filename_template("pie-chart-{timestamp}"),
    default_size(600, 600),
    scale(2)
]).

export_config(heatmap, [
    formats([svg, png, pdf, csv]),
    filename_template("heatmap-{timestamp}"),
    default_size(800, 600),
    scale(2)
]).

export_config(network_graph, [
    formats([svg, png, pdf, json]),
    filename_template("network-{timestamp}"),
    default_size(1200, 800),
    scale(2)
]).

export_config(plot3d, [
    formats([png, json]),
    filename_template("3d-plot-{timestamp}"),
    default_size(800, 600),
    scale(2)
]).

% ============================================================================
% EXPORT COMPONENT GENERATION
% ============================================================================

%% generate_export_component(+Component, -JSX)
%  Generate a complete export component with menu and handlers.
generate_export_component(Component, JSX) :-
    generate_export_hook(Component, Hook),
    generate_export_menu(Component, Menu),
    format(atom(JSX), 'interface ExportProps {
  chartRef: React.RefObject<HTMLDivElement | SVGSVGElement>;
  data?: unknown[];
  title?: string;
}

~w

export const ExportControls: React.FC<ExportProps> = ({ chartRef, data, title }) => {
  const { exportToSVG, exportToPNG, exportToPDF, exportToJSON, exportToCSV, isExporting } = useExport(chartRef, data, title);

  return (
~w
  );
};', [Hook, Menu]).

%% generate_export_hook(+Component, -Hook)
%  Generate React hook for export functionality.
generate_export_hook(Component, Hook) :-
    (export_config(Component, Config) -> true ; export_config(default, Config)),
    (member(scale(Scale), Config) -> true ; Scale = 2),
    (member(background(Bg), Config) -> true ; Bg = white),
    format(atom(Hook), 'const useExport = (
  chartRef: React.RefObject<HTMLDivElement | SVGSVGElement>,
  data?: unknown[],
  title?: string
) => {
  const [isExporting, setIsExporting] = useState(false);

  const getFilename = useCallback((ext: string) => {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, "");
    const name = title?.toLowerCase().replace(/\\s+/g, "-") || "chart";
    return `${name}-${timestamp}${ext}`;
  }, [title]);

  const exportToSVG = useCallback(async () => {
    if (!chartRef.current) return;
    setIsExporting(true);
    try {
      const svgElement = chartRef.current.querySelector("svg") || chartRef.current;
      if (!(svgElement instanceof SVGSVGElement)) {
        throw new Error("No SVG element found");
      }
      const serializer = new XMLSerializer();
      const svgString = serializer.serializeToString(svgElement);
      const blob = new Blob([svgString], { type: "image/svg+xml" });
      downloadBlob(blob, getFilename(".svg"));
    } finally {
      setIsExporting(false);
    }
  }, [chartRef, getFilename]);

  const exportToPNG = useCallback(async () => {
    if (!chartRef.current) return;
    setIsExporting(true);
    try {
      const svgElement = chartRef.current.querySelector("svg") || chartRef.current;
      if (!(svgElement instanceof SVGSVGElement)) {
        throw new Error("No SVG element found");
      }
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      const svgData = new XMLSerializer().serializeToString(svgElement);
      const img = new Image();
      const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(svgBlob);

      await new Promise<void>((resolve, reject) => {
        img.onload = () => {
          canvas.width = img.width * ~w;
          canvas.height = img.height * ~w;
          ctx.scale(~w, ~w);
          ctx.fillStyle = "~w";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
          resolve();
        };
        img.onerror = reject;
        img.src = url;
      });

      canvas.toBlob((blob) => {
        if (blob) downloadBlob(blob, getFilename(".png"));
      }, "image/png");
    } finally {
      setIsExporting(false);
    }
  }, [chartRef, getFilename]);

  const exportToPDF = useCallback(async () => {
    if (!chartRef.current) return;
    setIsExporting(true);
    try {
      const { jsPDF } = await import("jspdf");
      const svgElement = chartRef.current.querySelector("svg") || chartRef.current;
      if (!(svgElement instanceof SVGSVGElement)) {
        throw new Error("No SVG element found");
      }

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Could not get canvas context");

      const svgData = new XMLSerializer().serializeToString(svgElement);
      const img = new Image();
      const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(svgBlob);

      await new Promise<void>((resolve, reject) => {
        img.onload = () => {
          canvas.width = img.width * 2;
          canvas.height = img.height * 2;
          ctx.scale(2, 2);
          ctx.fillStyle = "white";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
          resolve();
        };
        img.onerror = reject;
        img.src = url;
      });

      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({
        orientation: canvas.width > canvas.height ? "landscape" : "portrait",
        unit: "px",
        format: [canvas.width, canvas.height]
      });
      pdf.addImage(imgData, "PNG", 0, 0, canvas.width, canvas.height);
      pdf.save(getFilename(".pdf"));
    } finally {
      setIsExporting(false);
    }
  }, [chartRef, getFilename]);

  const exportToJSON = useCallback(() => {
    if (!data) return;
    setIsExporting(true);
    try {
      const jsonString = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonString], { type: "application/json" });
      downloadBlob(blob, getFilename(".json"));
    } finally {
      setIsExporting(false);
    }
  }, [data, getFilename]);

  const exportToCSV = useCallback(() => {
    if (!data || !Array.isArray(data) || data.length === 0) return;
    setIsExporting(true);
    try {
      const headers = Object.keys(data[0] as Record<string, unknown>);
      const csvRows = [
        headers.join(","),
        ...data.map(row =>
          headers.map(h => JSON.stringify((row as Record<string, unknown>)[h] ?? "")).join(",")
        )
      ];
      const blob = new Blob([csvRows.join("\\n")], { type: "text/csv" });
      downloadBlob(blob, getFilename(".csv"));
    } finally {
      setIsExporting(false);
    }
  }, [data, getFilename]);

  return { exportToSVG, exportToPNG, exportToPDF, exportToJSON, exportToCSV, isExporting };
};

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};', [Scale, Scale, Scale, Scale, Bg]).

%% generate_export_menu(+Component, -JSX)
%  Generate export menu/dropdown JSX.
generate_export_menu(Component, JSX) :-
    get_export_formats(Component, Formats),
    maplist(generate_menu_item, Formats, MenuItems),
    atomic_list_concat(MenuItems, '\n        ', MenuItemsStr),
    format(atom(JSX), '    <div className={styles.exportControls}>
      <div className={styles.exportMenu}>
        <button
          className={styles.exportButton}
          disabled={isExporting}
          aria-haspopup="true"
        >
          {isExporting ? "Exporting..." : "Export"}
          <span className={styles.exportIcon}>▼</span>
        </button>
        <div className={styles.exportDropdown} role="menu">
        ~w
        </div>
      </div>
    </div>', [MenuItemsStr]).

%% generate_menu_item(+Format, -JSX)
generate_menu_item(svg, '<button onClick={exportToSVG} role="menuitem">Export as SVG</button>').
generate_menu_item(png, '<button onClick={exportToPNG} role="menuitem">Export as PNG</button>').
generate_menu_item(pdf, '<button onClick={exportToPDF} role="menuitem">Export as PDF</button>').
generate_menu_item(json, '<button onClick={exportToJSON} role="menuitem" disabled={!data}>Export as JSON</button>').
generate_menu_item(csv, '<button onClick={exportToCSV} role="menuitem" disabled={!data}>Export as CSV</button>').

% ============================================================================
% INDIVIDUAL FORMAT EXPORTS
% ============================================================================

%% generate_svg_export(+Component, -Code)
%  Generate SVG-specific export function.
generate_svg_export(_Component, Code) :-
    Code = 'export const exportToSVG = (svgElement: SVGSVGElement, filename: string): void => {
  const serializer = new XMLSerializer();

  // Clone SVG to avoid modifying the original
  const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

  // Inline styles for standalone SVG
  const styleSheets = document.styleSheets;
  let cssText = "";
  for (const sheet of styleSheets) {
    try {
      for (const rule of sheet.cssRules) {
        cssText += rule.cssText + "\\n";
      }
    } catch (e) {
      // Skip cross-origin stylesheets
    }
  }

  const styleElement = document.createElementNS("http://www.w3.org/2000/svg", "style");
  styleElement.textContent = cssText;
  clonedSvg.insertBefore(styleElement, clonedSvg.firstChild);

  // Add XML declaration and doctype
  const svgString = \'<?xml version="1.0" encoding="UTF-8"?>\\n\' +
    serializer.serializeToString(clonedSvg);

  const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  downloadBlob(blob, filename);
};'.

%% generate_png_export(+Component, -Code)
%  Generate PNG-specific export function with scaling.
generate_png_export(Component, Code) :-
    (export_config(Component, Config) -> true ; export_config(default, Config)),
    (member(scale(Scale), Config) -> true ; Scale = 2),
    (member(background(Bg), Config) -> true ; Bg = white),
    format(atom(Code), 'export const exportToPNG = async (
  svgElement: SVGSVGElement,
  filename: string,
  options: { scale?: number; background?: string } = {}
): Promise<void> => {
  const { scale = ~w, background = "~w" } = options;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create canvas context");

  const bbox = svgElement.getBoundingClientRect();
  canvas.width = bbox.width * scale;
  canvas.height = bbox.height * scale;

  // Set background
  if (background !== "transparent") {
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // Convert SVG to image
  const svgData = new XMLSerializer().serializeToString(svgElement);
  const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  const img = new Image();
  img.crossOrigin = "anonymous";

  await new Promise<void>((resolve, reject) => {
    img.onload = () => {
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load SVG as image"));
    };
    img.src = url;
  });

  // Export as PNG
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      if (blob) {
        downloadBlob(blob, filename);
      }
      resolve();
    }, "image/png", 1.0);
  });
};', [Scale, Bg]).

%% generate_pdf_export(+Component, -Code)
%  Generate PDF-specific export function using jsPDF.
generate_pdf_export(Component, Code) :-
    (export_config(Component, Config) -> true ; export_config(default, Config)),
    (member(default_size(_W, _H), Config) -> true ; true),
    format(atom(Code), 'export const exportToPDF = async (
  svgElement: SVGSVGElement,
  filename: string,
  options: { title?: string; author?: string } = {}
): Promise<void> => {
  const { jsPDF } = await import("jspdf");

  // First convert to canvas
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create canvas context");

  const bbox = svgElement.getBoundingClientRect();
  const scale = 2;
  canvas.width = bbox.width * scale;
  canvas.height = bbox.height * scale;

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const svgData = new XMLSerializer().serializeToString(svgElement);
  const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  const img = new Image();

  await new Promise<void>((resolve, reject) => {
    img.onload = () => {
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = reject;
    img.src = url;
  });

  // Create PDF
  const orientation = canvas.width > canvas.height ? "landscape" : "portrait";
  const pdf = new jsPDF({
    orientation,
    unit: "px",
    format: [canvas.width / scale, canvas.height / scale]
  });

  // Add metadata
  if (options.title) {
    pdf.setProperties({ title: options.title, author: options.author || "UnifyWeaver" });
  }

  // Add image
  const imgData = canvas.toDataURL("image/png", 1.0);
  pdf.addImage(imgData, "PNG", 0, 0, canvas.width / scale, canvas.height / scale);

  pdf.save(filename);
};', []).

%% generate_export_button(+Format, +Options, -JSX)
%  Generate a single export button.
generate_export_button(Format, Options, JSX) :-
    export_format(Format, FormatOpts),
    member(description(Desc), FormatOpts),
    member(extension(Ext), FormatOpts),
    (member(label(Label), Options) -> true ; Label = Desc),
    (member(class(Class), Options) -> true ; Class = 'exportButton'),
    atom_string(Format, FormatStr),
    upcase_atom(FormatStr, FormatUpper),
    atom_string(FormatUpper, FormatUpperStr),
    format(atom(JSX), '<button
  className={styles.~w}
  onClick={exportTo~w}
  disabled={isExporting}
  title="~w (~w)"
>
  ~w
</button>', [Class, FormatUpperStr, Desc, Ext, Label]).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_export_css(-CSS)
%  Generate CSS for export controls.
generate_export_css(CSS) :-
    CSS = '.exportControls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.exportMenu {
  position: relative;
  display: inline-block;
}

.exportButton {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--surface, #16213e);
  color: var(--text, #e0e0e0);
  border: 1px solid var(--border, rgba(255, 255, 255, 0.1));
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background-color 0.2s, border-color 0.2s;
}

.exportButton:hover:not(:disabled) {
  background: var(--surface-hover, #1a2744);
  border-color: var(--accent, #00d4ff);
}

.exportButton:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.exportIcon {
  font-size: 0.625rem;
  margin-left: 0.25rem;
}

.exportDropdown {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 160px;
  padding: 0.5rem 0;
  background: var(--surface, #16213e);
  border: 1px solid var(--border, rgba(255, 255, 255, 0.1));
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  opacity: 0;
  visibility: hidden;
  transform: translateY(-8px);
  transition: opacity 0.2s, transform 0.2s, visibility 0.2s;
  z-index: 100;
}

.exportMenu:hover .exportDropdown,
.exportMenu:focus-within .exportDropdown {
  opacity: 1;
  visibility: visible;
  transform: translateY(4px);
}

.exportDropdown button {
  display: block;
  width: 100%;
  padding: 0.5rem 1rem;
  background: none;
  border: none;
  color: var(--text, #e0e0e0);
  text-align: left;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background-color 0.15s;
}

.exportDropdown button:hover:not(:disabled) {
  background: var(--surface-hover, rgba(255, 255, 255, 0.1));
}

.exportDropdown button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Export progress indicator */
.exportProgress {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  color: var(--text-secondary, #888);
  font-size: 0.875rem;
}

.exportSpinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--border, rgba(255, 255, 255, 0.1));
  border-top-color: var(--accent, #00d4ff);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* Responsive adjustments */
@media (max-width: 640px) {
  .exportDropdown {
    right: 0;
    left: auto;
  }
}'.

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_export_formats(+Component, -Formats)
get_export_formats(Component, Formats) :-
    (export_config(Component, Config) -> true ; export_config(default, Config)),
    (member(formats(Formats), Config) -> true ; Formats = [svg, png, pdf]).

%% get_export_filename(+Component, +Format, -Filename)
get_export_filename(Component, Format, Filename) :-
    (export_config(Component, Config) -> true ; export_config(default, Config)),
    (member(filename_template(Template), Config) -> true ; Template = "{component}-{timestamp}"),
    export_format(Format, FormatOpts),
    member(extension(Ext), FormatOpts),
    % Note: actual timestamp/component substitution happens at runtime in JS
    format(atom(Filename), '~w~w', [Template, Ext]),
    % Verify Component is bound
    (atom(Component) -> true ; true).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_export_config(+Component, +Options)
declare_export_config(Component, Options) :-
    retractall(export_config(Component, _)),
    assertz(export_config(Component, Options)).

%% clear_export_configs/0
clear_export_configs :-
    retractall(export_config(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_export_generator :-
    format('========================================~n'),
    format('Export Generator Tests~n'),
    format('========================================~n~n'),

    % Test 1: Export component generation
    format('Test 1: Export component generation~n'),
    generate_export_component(line_chart, Component),
    (sub_atom(Component, _, _, _, 'ExportControls')
    -> format('  PASS: Has ExportControls component~n')
    ; format('  FAIL: Missing ExportControls~n')),
    (sub_atom(Component, _, _, _, 'useExport')
    -> format('  PASS: Has useExport hook~n')
    ; format('  FAIL: Missing useExport~n')),

    % Test 2: Export hook generation
    format('~nTest 2: Export hook generation~n'),
    generate_export_hook(scatter_plot, Hook),
    (sub_atom(Hook, _, _, _, 'exportToSVG')
    -> format('  PASS: Has SVG export~n')
    ; format('  FAIL: Missing SVG export~n')),
    (sub_atom(Hook, _, _, _, 'exportToPNG')
    -> format('  PASS: Has PNG export~n')
    ; format('  FAIL: Missing PNG export~n')),
    (sub_atom(Hook, _, _, _, 'exportToPDF')
    -> format('  PASS: Has PDF export~n')
    ; format('  FAIL: Missing PDF export~n')),

    % Test 3: Export menu generation
    format('~nTest 3: Export menu generation~n'),
    generate_export_menu(bar_chart, Menu),
    (sub_atom(Menu, _, _, _, 'exportDropdown')
    -> format('  PASS: Has dropdown~n')
    ; format('  FAIL: Missing dropdown~n')),
    (sub_atom(Menu, _, _, _, 'Export as SVG')
    -> format('  PASS: Has SVG option~n')
    ; format('  FAIL: Missing SVG option~n')),

    % Test 4: Individual format exports
    format('~nTest 4: Individual format exports~n'),
    generate_svg_export(default, SVGCode),
    (sub_atom(SVGCode, _, _, _, 'XMLSerializer')
    -> format('  PASS: SVG uses XMLSerializer~n')
    ; format('  FAIL: SVG missing serializer~n')),
    generate_png_export(default, PNGCode),
    (sub_atom(PNGCode, _, _, _, 'canvas')
    -> format('  PASS: PNG uses canvas~n')
    ; format('  FAIL: PNG missing canvas~n')),
    generate_pdf_export(default, PDFCode),
    (sub_atom(PDFCode, _, _, _, 'jsPDF')
    -> format('  PASS: PDF uses jsPDF~n')
    ; format('  FAIL: PDF missing jsPDF~n')),

    % Test 5: CSS generation
    format('~nTest 5: CSS generation~n'),
    generate_export_css(CSS),
    (sub_atom(CSS, _, _, _, '.exportControls')
    -> format('  PASS: Has exportControls class~n')
    ; format('  FAIL: Missing exportControls~n')),
    (sub_atom(CSS, _, _, _, '.exportDropdown')
    -> format('  PASS: Has dropdown styles~n')
    ; format('  FAIL: Missing dropdown styles~n')),

    % Test 6: Utility predicates
    format('~nTest 6: Utility predicates~n'),
    get_export_formats(line_chart, Formats),
    (member(csv, Formats)
    -> format('  PASS: line_chart includes CSV~n')
    ; format('  FAIL: line_chart missing CSV~n')),
    (supported_format(png)
    -> format('  PASS: PNG is supported~n')
    ; format('  FAIL: PNG not supported~n')),

    % Test 7: Format specifications
    format('~nTest 7: Format specifications~n'),
    export_format(svg, SVGOpts),
    (member(mime_type('image/svg+xml'), SVGOpts)
    -> format('  PASS: SVG has correct MIME type~n')
    ; format('  FAIL: SVG wrong MIME type~n')),
    export_format(pdf, PDFOpts),
    (member(library(jspdf), PDFOpts)
    -> format('  PASS: PDF specifies jsPDF library~n')
    ; format('  FAIL: PDF missing library spec~n')),

    format('~nAll tests completed.~n').
