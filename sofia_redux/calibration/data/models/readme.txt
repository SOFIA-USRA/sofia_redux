
general parameters used for the models:

Diam      : Average angular diameter of the planet 
Dg        : geocentric distance
<R1bar>   : average radius seen at P=1bar or surface (vary with sub-earth-point latitude)
Compounds : used in the simulation  

Planete      <sep> <R1bar>	   <Dg>       Diam	Compounds
                   km		   km	      (")
Mars	     0	    3386.2  	1.746031E+08  8.0000    co,h2o,isotopes (Fouchet lat+45)
Jupiter	     0	   69134.1	7.129966E+08 40.0000    co,hcn,cs,h2o(hybe),ch4,ph3,nh3
Saturn	     0	   57239.9	1.311842E+09 18.0000    h2o,ch4,ph3,nh3
Uranus	     0	   25264.3	2.977792E+09  3.5000	h2o,ch4,nh3
Neptune	     28	   24599.8	4.412237E+09  2.3000	co,hcn,h2o,ch4,nh3

Callisto     0	    2409.3	7.129966E+08  1.3940	eps = 1.2, r=0, Spencer 2L
Ganymede     0	    2632.3	7.129966E+08  1.5230	eps = 3.1, r=0, Spencer 2L
Europa	     0	    1560.8	7.376671E+08  0.8728	eps = 1.0, r=0, Spencer 2L
Io	     0	    1821.3	7.376671E+08  1.0185	eps = 1.0, r=0, Spencer 2L
Titan	     0	    2575.0	1.311842E+09  0.8097	co,hcn,h2o,ch4,hc3n,ch3cn


Spectral resolution:
	 The files with "*_1*" has a spectral resolution of 1 MHz.
	 The files with "*_2*" has a spectral resolution of 1 GHz.


Unit of the Output file:
      N	    Freq     Tb       Flux        Trj 
	    (GHz)    (K)      (Jy)	  (K)
      1     2.00000  361.4608 0.1313E+01  361.4127
      2     3.00000  295.1912 0.2410E+01  295.1192
      3     4.00000  273.3416 0.3967E+01  273.2456
      4     5.00000  253.8520 0.5756E+01  253.7319
  
Freq is the Frequency 
Tb   is the Brightness Temperature
Flux is the total Flux density
Trj  is the Rayleigh-Jeans temperature
