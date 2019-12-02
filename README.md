# Monte Carlo Ray Tracing Engine Built in C++

The engine uses a Monte Carlo technique to render images. This process randomly fires rays on contact with Lambertian materials. This creates much more accurate lighting than traditional Ray Tracing. The downside is of course speed.

## Motivation

This engine started out as a required school project, but quickly developed further than the requirements.

## Tech/framework used

Built with:  

[C++] () 
[CMake] (https://www.python.org)
[Open MP] ()

## Features
What makes your project stand out?

## Example Images
Show what the library does as concisely as possible, developers should be able to figure out how your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation

### Building the Executable

After downloading the repository, navigate to the directory where the repository is stored.

From the Command Line, Enter The Command

``` bash
cmake . && make
```

### Running the Engine:

After You Have Built the Executable, Run The Program Using the Command 

``` bash
./raytracer [Driver File] [Ouput File] [Samples]
```

The output format for the image is the PPM model. You can open such images in most image viewers and applications such as Adobe Photoshop. The number of Samples directly coresponds to the amount of noise in the image. The higher the samples, the higher quality of an image.

### Creating Driver Files

For your convenience, example driver files have been provided under the folder titled ExampleDriverFiles. These driver
files are all fully featured to create interesting and dramatic images that showcase the features of the engine.

If you wish to create your own driver files, the following fields are required for each element:

eye X Y Z  
look X Y Z  
up X Y Z  
d Distance  
bounds Left Right Bottom Top  
res Width Height  
sphere X Y Z Radius AlbedoRed AlbedoGreen AlbedoBlue  
model RotationX RotationY RotationZ RotationTheta ScaleFactor TranslationX TranslationY TranslationZ SmoothingTheta FilePath

For Spheres and Model Material Files, optional fields exist. Appending "light", "mirror", or "glass" to the end of a sphere line or material description will activate the corresponding property for that section of the image. See reference driver files for examples.

## Where to Find OBJ Files

Finding obj files is inconsequential. Simply look up ".obj files" on the internet and choose one that fits your liking.
Make sure that you create the proper material file using the format showcases in the example files.

## Created by Max Rosoff with the advice of Ben Sattelberg and Ross Beveridge
