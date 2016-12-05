# solid_hydrogen
a collection of scripts for solid hydrogen research
1. static_twists.py
  collect all QMCPACK scalar data from a directory, which may contain one or many subdirectories of VMC/DMC runs.
2. forces.py
  functions related to extracting and analyzing the forces of a QMCPACK run
3. generate_confis.py 
  pulls atomic configurations from a QE output and puts them in a database in json format
4. static_correlation.py reads atomic configurations from a database in json format and calculates g(r) and S(k)

