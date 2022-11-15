To make all HTML docs:

```
   make clean
   make html
```
   
To make all manual PDFs:
```
   cd manuals/exes/users/
   make clean
   make latexpdf
   cd ../data_handbook
   make clean
   make latexpdf
   cd ../developers
   make clean
   make latexpdf

   cd manuals/fifils/users/
   make clean
   make latexpdf
   cd ../data_handbook
   make clean
   make latexpdf
   cd ../developers
   make clean
   make latexpdf

   cd ../../flitecam/users/
   make clean
   make latexpdf
   cd ../data_handbook
   make clean
   make latexpdf
   cd ../developers
   make clean
   make latexpdf

   cd ../../forcast/users/
   make clean
   make latexpdf
   cd ../data_handbook
   make clean
   make latexpdf
   cd ../developers
   make clean
   make latexpdf

   cd ../../hawc/users/
   make clean
   make latexpdf
   cd ../data_handbook
   make clean
   make latexpdf
   cd ../developers
   make clean
   make latexpdf
```