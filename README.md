# Monte Carlo Raytracing Engine

### A Dynamic Fully Fledged Raytracing Engine Built in C++

# About Project

An engine built to render complex scenes with all the capabilities of real world private engines. Images are outputted in the PPM format and use a simple driver file system for creation.

The engine uses a Monte Carlo technique to render images. This process randomly fires rays on contact with Lambertian materials. This creates much more accurate lighting than traditional Ray Tracing. The downside is of course speed. The engine can handle simple Lambertian surfaces, metal surfaces with total reflection, and refractive surfaces such as glass. The engine also produces accurate shadows and highlights and has complex material control.

Current work on this project is to migrate all base files to GPU compatible Cuda code.

## Example Images

Shown below are the resulting images when running the engine with the driver "Spheres".

| Number of Samples | Resulting Image |
| ----------------- | --------------- |
| 10 Samples | ![10 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres10.jpg) |
| 100 Samples | ![100 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres100.jpg) |
| 1000 Samples | ![1000 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres1000.jpg) |
| 10000 Samples | ![10000 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres10000.jpg) |

Shown below are the resulting images when running the engine with the driver "Mando".

| Number of Samples | Resulting Image |
| ----------------- | --------------- |
| 10 Samples | ![10 Samples](./ExampleDriverFiles/Mando/Images/JPG/Mando10.jpg) |
| 100 Samples | ![100 Samples](./ExampleDriverFiles/Mando/Images/JPG/Mando100.jpg) |

## Installation

### Building the Executable

After downloading the repository, navigate to the directory where the repository is stored.

From the Command Line, Enter The Command

``` bash
cmake . && make
```

### Running the Engine

After You Have Built the Executable, Run The Program Using the Command 

``` bash
./raytracer [Driver File] [Ouput File] [Samples]
```

The output format for the image is the PPM model. You can open such images in most image viewers and applications such as Adobe Photoshop. The number of Samples directly coresponds to the amount of noise in the image. The higher the samples, the higher quality of an image.

### Node Driver

In addition, you can build this project as a Node.js package. Simply install the package, and follow the below documentation.

```bash
npm install
```

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
model RX RY RZ RTheta ScaleFactor TX TY TZ SmoothingTheta FilePath

For Spheres and for Model Material Files, optional fields exist. Appending "light", "mirror", or "glass" to the end of a sphere line or material description will activate the corresponding property for that section of the image. See reference driver files for examples. Make sure that the bounds and resolution share an aspect ratio. This is important to ensure no artifacts.

## Where to Find OBJ Files

Finding obj files is inconsequential. Simply look up ".obj files" on the internet and choose one that fits your liking.
Make sure that you create the proper material file using the format showcases in the example files. Files with Triangle counts of over 10K will result in images that take a significant time to render.
