#!/bin/perl
=pod
Generate C/C++ code from OpenCL files corresponding to kernel modules.

Copyright (C) 2020 Qijia (Michael) Jin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
=cut
use strict;
use warnings;

die "$0: error: please give filename to parse!\n" if ($#ARGV == -1);

my $fd = undef;

open ($fd, $ARGV[0])
	|| die "$0: error: unable to open $ARGV[0]: $!";

printf("/* %s
 *
 * Copyright (C) 2020 Qijia (Michael) Jin
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

const unsigned char %s[] = {\n\t", $ARGV[0] =~ s/\.opencl//gr, $ARGV[0] =~ s/\.opencl/_opencl/gr);

my $i = 0;

while (!eof($fd)) {
	my $c = getc($fd);
	#printf(", 0x%02x", unpack('C', $c));
	printf("0x%02x, ", unpack('C', $c));

	++$i;
	if ($i % 8 == 0) {
		print "\n\t";
	}
}

# add null character to end of array
if ($i % 8 == 0) {
	print "0x00, \n\t";
}
else {
	print "0x00, ";
}

print "\n};\n\n";

close ($fd)
	|| warn "$0: error: unable to close $ARGV[0]: $!"
