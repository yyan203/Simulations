#!/usr/bin/perl
use Shell;

# for converting indent-stress-dist to .data file

if($#ARGV < 1) {
  print "Correct syntax: stress2xyz-version2.pl file element_size\n";
  exit;
}

foreach (@ARGV) {
  lmps2data($_);
}

sub lmps2data(){

my $output=0;#flag for output to datafile or not
my $logfile=$ARGV[0];
  print "$logfile\n";
my $size=$ARGV[1];
  print "$size\n";
my $datafile="$logfile.xyz";
my $latafile="log.lammps";

# check if log.lammps file exists

my $z;
if(-e $latafile){
  print "log.lammps exist!\n";
  open(LATA, $latafile);
  my $count_line=0;

  while(<LATA>){
  chomp;
  $count_line++;
  if($count_line==42){
  my @line=split(' ',$_);
  $z=$line[9];
  chop($z);
  print "z thickness is $z\n";
  last;
  }
  if($count_line>42) {close(LATA);exit;}
  }
  close(LATA);

  } else { print "log.lammps doesn't exist! Use default thickness 10.3581\n"; $z=10.3581;}


my $element_V=$z*$size*$size;

#print "logfile is $logfile\n";
#print "datafile is $datafile\n";

print "logfile not exist!\n"
  unless(-e $logfile);
if(-e $datafile) {
  print "datafile exist!\n";}


open(LOG, $logfile);
open(DATA, ">$datafile");
while(<LOG>) {
  chomp;
  #if(/^Loop /) {$output=0;}
  if(/^\w/) {
    my @title=split(' ', $_);
    $num=$title[1];
    $step=$title[0];
    print DATA "$num" , "\n", "TIMESTEP ", "$step","\n";
  }
  if(/^\s/){
    my @title=split(' ', $_);
    $num_atom=$title[4];
    if($num_atom>0)
    #{print DATA "$num_atom $title[0] $title[1] $title[2] $title[3] $title[4]\n";}
    #{print DATA "$num_atom $_\n";}
    #printf("%.2f", 6.02 * 1.25) = 7.52
    {printf(DATA "%d %d %d %.2f %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",$title[0],$title[1],$title[2],$title[3],$title[4],$title[5]*$title[4]/$element_V,$title[6]*$title[4]/$element_V,$title[7]*$title[4]/$element_V,$title[8]*$title[4]/$element_V,$title[9]*$title[4]/$element_V,$title[10]*$title[4]/$element_V,($title[5]+$title[6]+$title[7])*$title[4]/$element_V);}
    else {print DATA "$title[0] 0 0 0 0 0 0 0 0 0 0 10\n";}
    }
}
close(LOG);
close(DATA);
}
