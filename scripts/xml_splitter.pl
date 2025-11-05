#!/usr/bin/env perl
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (@s243a)
#
# xml_splitter.pl - Splits xmllint output and optionally repairs namespaces.

use strict;
use warnings;

if (@ARGV < 2) {
    die "Usage: $0 <repair_flag> <namespace_map_string> <tag1> <tag2> ...\n";
}

my $repair = shift @ARGV;
my $ns_map_str = shift @ARGV;
my @tags = @ARGV;

my %ns_map;
if ($ns_map_str) {
    my @ns_pairs = split /;/, $ns_map_str;
    for my $pair (@ns_pairs) {
        my ($pfx, $uri) = split /=/, $pair, 2;
        $ns_map{$pfx} = $uri if $pfx && $uri;
    }
}

local $/;
my $data = <STDIN>;

my $pattern = join '|', map { quotemeta } @tags;
while ($data =~ m{(<($pattern)(?:\s+[^>]*)?>.*?</\2>)}gs) {
    my $record = $1;

    if ($repair eq 'true') {
        my $first_gt = index($record, ">");
        if ($first_gt != -1) {
            my $header = substr($record, 0, $first_gt);
            my %prefixes;
            while ($record =~ /[<\s]([A-Za-z_][\w.-]*):/g) {
                $prefixes{$1} = 1;
            }

            my @needed;
            for my $prefix (sort keys %prefixes) {
                next if $prefix eq 'xml';
                my $uri = $ns_map{$prefix};
                next unless $uri;

                unless ($header =~ /xmlns:$prefix=/) {
                    push @needed, "xmlns:$prefix=\"$uri\"";
                }
            }

            if (@needed) {
                my $insert_pos = index($record, ' ');
                if ($insert_pos == -1 || $insert_pos > $first_gt) {
                    $insert_pos = $first_gt;
                }
                my $insertion = ' ' . join(' ', @needed);
                substr($record, $insert_pos, 0, $insertion);
            }
        }
    }

    print $record, "\0";
}
