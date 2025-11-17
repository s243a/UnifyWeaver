#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use File::Find;
use JSON::PP ();

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
    my $content = slurp_file($filepath);
    my ($front_matter, $body) = split_front_matter($content);
    return unless front_matter_matches($front_matter);
    parse_and_print_records($body, $filepath);
}

sub process_stdin {
    my $content = do { local $/; <STDIN> };
    my ($front_matter, $body) = split_front_matter($content // '');
    return unless front_matter_matches($front_matter);
    parse_and_print_records($body, 'STDIN');
}

sub parse_and_print_records {
    my ($content, $filepath) = @_;
    my @lines = split /\n/, $content;

    my $in_record = 0;
    my $in_metadata = 0;
    my $in_code = 0;
    my $record_content = '';
    my %metadata = ();
    my $code_content = '';
    my $header = '';

    for my $line (@lines) {
        if ($line =~ /^###?\s*(.*)/) {
            # We found a new record header (## or ###), process the previous one if it exists
            if ($in_record) {
                process_found_record(\%metadata, $header, $code_content, $record_content);
            }
            # Reset for the new record
            $in_record = 1;
            $in_metadata = 0;
            $in_code = 0;
            $record_content = "$line\n";
            %metadata = ();
            $code_content = '';
            $header = $1;
        } elsif ($in_record) {
            $record_content .= "$line\n";
            if ($line =~ /^>\s*\[!example-record\]/) {
                $in_metadata = 1;
            } elsif ($in_metadata && $line =~ /^>\s*(\w+):\s*(.*)/) {
                $metadata{$1} = $2;
            } elsif ($in_metadata && $line !~ /^>/) {
                $in_metadata = 0; # End of metadata block
            } elsif ($line =~ /^```/) {
                $in_code = !$in_code; # Toggle in/out of code block
            } elsif ($in_code) {
                $code_content .= "$line\n";
            }
        }
    }
    # Process the last record if it exists
    if ($in_record) {
        process_found_record(\%metadata, $header, $code_content, $record_content);
    }
}

sub process_found_record {
    my ($metadata_ref, $header, $code, $full_content) = @_;
    my %metadata = %{$metadata_ref};
    $metadata{'header'} = $header;

    # Apply query filter (check both 'id' and 'name' fields)
    if ($query) {
        my $matches = 0;
        $matches = 1 if $metadata{'id'} && $metadata{'id'} =~ /$query/;
        $matches = 1 if $metadata{'name'} && $metadata{'name'} =~ /$query/;
        return unless $matches;
    }

    # Format and print
    if ($format eq 'content') {
        print $code;
    } elsif ($format eq 'json') {
        $metadata{'content'} = $code;
        print JSON::PP->new->pretty->encode(\%metadata);
    } else { # 'full'
        print $full_content;
    }
    print $separator;
}

sub split_front_matter {
    my ($content) = @_;
    my @lines = split /\n/, $content, -1;
    return ({}, $content) unless @lines && $lines[0] =~ /^---\s*$/;

    shift @lines; # drop opening ---
    my @front_lines;
    while (@lines) {
        my $line = shift @lines;
        last if $line =~ /^---\s*$/;
        push @front_lines, $line;
    }
    my $body = join("\n", @lines);

    my %front;
    for my $line (@front_lines) {
        $line =~ s/\r$//;
        next unless $line =~ /\S/;
        my ($k, $v) = split /:\s*/, $line, 2;
        $k =~ s/\s+$// if defined $k;
        $v =~ s/\s+$// if defined $v;
        $front{$k} = defined $v ? $v : '';
    }

    return (\%front, $body);
}

sub front_matter_matches {
    my ($front) = @_;
    return 1 unless defined $file_filter && length $file_filter && $file_filter ne 'all';
    my @clauses = split /\s*,\s*/, $file_filter;
    foreach my $clause (@clauses) {
        next unless length $clause;
        my ($k, $v) = split /=/, $clause, 2;
        for ($k, $v) { $_ = '' unless defined $_; s/^\s+|\s+$//g; }
        return 0 unless exists $front->{$k} && $front->{$k} eq $v;
    }
    return 1;
}

sub slurp_file {
    my ($filepath) = @_;
    open my $fh, '<:encoding(UTF-8)', $filepath
        or die "Cannot open $filepath: $!";
    local $/;
    my $content = <$fh>;
    close $fh;
    return $content;
}

exit 0;
