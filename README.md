# Monte Carlo Ray Tracing Engine Built in C++

FOR BEN AND WIN: THE DRIVER FILE THAT I AM SUBMITTING FOR A4 is CALLED WackyRefractedSpheres
YOU CAN RUN IT BY RUNNING ./raytracer ExampleDriverFiles/WackyRefractedSpheres/WackyRefractedSpheres.txt ./image.ppm

### Building the Executable

After downloading the repository, navigate to the directory where the repository is stored.

From the Command Line, Enter The Command

``` bash
cmake . && make
```

### Running the Engine:

After You Have Built the Executable, Run The Program Using the Command 

``` bash
./raytracer [driverFile] [imageOutputFile] [numSamples (Optional)]
```

Inputting a number of samples is optional and will run the engine in "Monte Carlo Mode". This computation is
significantly more complex and the user should be prepared to wait quite a while for completion.

A progress meter will be displayed at intervals of 10%. If the engine runs for longer than 3 hours and you are running
one of the Example Driver Files with a sample number of less than 10000, an error has occurred. Please make an issue 
where you link the driver files and the run configurations you used and I will look into the the error.

### Creating Driver Files

For your convenience, example driver files have been provided under the folder titled ExampleDriverFiles. These driver
files are all fully featured to create interesting and dramatic images that showcase the features of the engine.
Those folders marked with the starting letters MC are intended for Monte Carlo Ray Tracing.

If you wish to create your own driver files, the following fields are required for each element:

eye x y z  
look x y z  
up x y z  
d distance  
bounds left right bottom top  
res width height  
recursionlevel depth  
ambient r g b  
light x y z 1 0.5 0.5 0.5  
model rx ry rz theta scale tx ty tz smoothTheta objFilePath

### Where to Find .obj Files

Finding obj files is inconsequential. Simply look up .obj files on the internet and choose one that fits your liking.
Make sure that you create the proper material file using the format showcases in the example files.

### Created by Max Rosoff with the advice of Ben Sattelberg and Ross Beveridge
