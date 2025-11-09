#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use File::Find;
use Path::Tiny;

# =============================================================================
# Tool: extract_records.pl
#
# A stream-oriented tool that parses Markdown files conforming to the
# "Example Record Format" and outputs the records.
#
# Specification: docs/development/EXTRACT_RECORDS_TOOL_SPECIFICATION.md
# =============================================================================

# --- Default Configuration ---
my $separator = "\0";
my $format = 'full';
my $query = '';
my $file_filter = 'file_type=UnifyWeaver Example Library';
my $help = 0;

# --- CLI Argument Parsing ---
GetOptions(
    's|separator=s' => \$separator,
    'f|format=s'    => \$format,
    'q|query=s'     => \$query,
    'file-filter=s' => \$file_filter,
    'h|help'        => \$help,
) or die "Error in command line arguments\n";

if ($help) {
    print "Usage: $0 [OPTIONS] [PATH...]\n";
    # TODO: Add full help text here
    exit 0;
}

# --- Main Logic ---

# 1. Build a list of files to process
my @files_to_process;
if (@ARGV) {
    find(
        sub {
            return unless -f;
            return unless /\.md$|\.markdown$/;
            push @files_to_process, $File::Find::name;
        },
        @ARGV
    );
} else {
    # Reading from STDIN, no file list
}

# 2. Process files or STDIN
if (@files_to_process) {
    foreach my $file (@files_to_process) {
        process_file($file);
    }
} else {
    process_stdin();
}

# --- Subroutines ---

sub process_file {
    my ($filepath) = @_;
    # TODO:
    # 1. Read file content
    # 2. Check for YAML frontmatter
    # 3. Apply --file-filter
    # 4. If valid, call parse_and_print_records()
    print "# Processing file: $filepath\n"; # Placeholder
}

sub process_stdin {
    # TODO:
    # 1. Read content from STDIN
    # 2. Call parse_and_print_records()
    print "# Processing STDIN\n"; # Placeholder
}

sub parse_and_print_records {
    my ($content, $filepath) = @_;
    # TODO:
    # 1. Find all record headers (###)
    # 2. For each record:
    #    a. Parse metadata callout
    #    b. Apply --query filter to 'name'
    #    c. Extract content block
    #    d. Format output based on --format
    #    e. Print formatted output followed by separator
}

exit 0;
