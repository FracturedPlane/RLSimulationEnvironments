

##Dependancies:

 1. sudo pip install pyOpenGL
 1. sudo apt-get install libode-dev
 1. sudo pip install pyODE


## Install

Follow instructions for building on Linux (Ubuntu)
 1.  Create a variable for the path to the TerrainRL folder  
  ```export RLSIMENV_PATH=/path/to/terrainRL``` 
  or
   ```setenv RLSIMENV_PATH /path/to/terrainRL``` 
   
```
pip3 install --user -v -e ./
```

### Build Off-Screen rendering

For some of the environments (projectileGame) visual data is used from the OpenGL simulation.
You will need to compile this rendering backend for these environments to function 

```
cd rendering
premake4 gmake
make config=release64 -j 4
cd ../
```
