# Monte Carlo Ray Tracing Engine Built in C++

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

### Where to Find OBJ Files

Finding obj files is inconsequential. Simply look up ".obj files" on the internet and choose one that fits your liking.
Make sure that you create the proper material file using the format showcases in the example files.

### Created by Max Rosoff with the advice of Ben Sattelberg and Ross Beveridge









Project title
A little info about your project and/ or overview that explains what the project is about.

Motivation
A short description of the motivation behind the creation and maintenance of the project. This should explain why the project exists.

Build status
Build status of continus integration i.e. travis, appveyor etc. Ex. -

Build Status Windows Build Status

Code style
If you're using any code style like xo, standard etc. That will help others while contributing to your project. Ex. -

js-standard-style

Tech/framework used
Ex. -

Built with

Electron
[Python] (https://www.python.org)
Features
What makes your project stand out?

Code Example
Show what the library does as concisely as possible, developers should be able to figure out how your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

Installation
Provide step by step series of examples and explanations about how to get a development env running.

API Reference
Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

Tests
Describe and show how to run the tests with code examples.

How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.
