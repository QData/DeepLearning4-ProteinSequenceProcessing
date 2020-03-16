RELATED SOFTWARE INSTALLATION:

---- install BLAST from ncbi (NR filtered db)

http://www.ncbi.nlm.nih.gov/staff/tao/URLAPI/blastpgp.html#1.1

setup using instrusction from : http://www.ncbi.nlm.nih.gov/staff/tao/URLAPI/unix_setup.html ftp://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/

BLAST database is a key component of any BLAST search. Here we need to first download the NR database ==> ftp ftp.ncbi.nlm.nih.gov using "anonymous" ,
then " cd blast/db ", then "get nr.*tar.gz" , then "tar zxvpf nr.*tar.gz" ==> nr.*tar.gz (non-redundant protein sequence database with entries from GenPept, Swissprot, PIR, PDF, PDB, and NCBI RefSeq )

Configuration: http://www.ncbi.nlm.nih.gov/staff/tao/URLAPI/blastpgp.html#3.2

We can further filter nr database to nrf database using softwares provided by PSIPred

This step is optional !

Software PSIPred : http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/README

A program called "pfilt" is included which will filter FASTA files before using the formatdb command to generate the encoded BLAST data bank files.

For example: pfilt nr.fasta > nrfilt formatdb -t nrfilt -i nrfilt cp nrfilt.p?? $BLASTDB

(The above command assumes you have already set the BLASTDB environment variable to the directory where you usually keep your BLAST data banks)

