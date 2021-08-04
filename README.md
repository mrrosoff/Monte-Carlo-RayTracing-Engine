# Monte Carlo Raytracing Engine

### A Dynamic Fully Fledged Raytracing Engine Built in C++

# About Project

An engine built to render complex scenes with all the capabilities of real world private engines. Images are outputted in the PPM format and use a simple driver file system for creation.

The engine uses a Monte Carlo technique to render images. This process randomly fires rays on contact with Lambertian materials. This creates much more accurate lighting than traditional Ray Tracing. The downside is of course speed. The engine can handle simple Lambertian surfaces, metal surfaces with total reflection, and refractive surfaces such as glass. The engine also produces accurate shadows and highlights and has complex material control.

Current work on this project is to migrate all base files to GPU compatible Cuda code.

## Example Images

### Spheres

Shown below are the resulting images when running the engine with the driver "Spheres".

| Number of Samples | Resulting Image |
| ----------------- | --------------- |
| 10 Samples | ![10 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres10.jpg) |
| 100 Samples | ![100 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres100.jpg) |
| 1000 Samples | ![1000 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres1000.jpg) |
| 10000 Samples | ![10000 Samples](./ExampleDriverFiles/Spheres/Images/JPG/Spheres10000.jpg) |

### Mando

Shown below are the resulting images when running the engine with the driver "Mando".

| Number of Samples | Resulting Image |
| ----------------- | --------------- |
| 10 Samples | ![10 Samples](./ExampleDriverFiles/Mando/Images/JPG/Mando10.jpg) |
| 100 Samples | ![100 Samples](./ExampleDriverFiles/Mando/Images/JPG/Mando100.jpg) |

## Installation

### Building the Executable

After downloading the repository, navigate to the directory where the repository is stored.

From the command line, enter the following commands.

``` bash
mkdir build
cmake .. && make
```

At this time, we do not support Windows or MacOS.

### Running the Engine

After You Have Built the Executable, Run The Program Using the Command 

``` bash
./raytracer [Driver File] [Ouput File] [Samples]
```

The output format for the image is the PPM model. You can open such images in most image viewers and applications such as Adobe Photoshop. The number of Samples directly coresponds to the amount of noise in the image. The higher the samples, the higher quality of an image.

### Node Driver

In addition, you can build this project as a Node.js package. Simply install the package with the following command.

```bash
npm install raytracer-node
```

Here are a couple of example functions that utilize the library (written with ES6 syntax). Please note that node is single threaded, and as such, this library will appear to "crash" your thread. It will complete its task given time. If you wish to use a rendering engine with this program, I would recommend looking into experimental node features or using worker windows in something like electron to hide the slowdown.

```javascript
function runRaytracer() {
    createImage(
        [
            "0 0.25 -30", 
            "0 0 1", 
            "0 1 0", 
            "-10", 
            "-10 10 -10 10", 
            "400 400", 
            "100"
        ],
		[
			"0 0 -10000000030 10000000000 0.3 0.3 0.8",
			"0 0 10000000020 10000000000 0.3 0.3 0.8",
			"0 -10000000020 0 10000000000 0.9 0.9 0.9",
			"0 10000000020 0 10000000000 0.9 0.9 0.9",
			"0 59.5 0 40 1.5 1.5 1.5 light",
			"-10000000020 0 0 10000000000 0.3 0.8 0.3",
			"10000000020 0 0 10000000000 0.8 0.3 0.3",
			"-7 -17.1 0 3 0.8 0.8 0.8 glass",
			"5 -17.1 5 3 0.8 0.8 0.8 mirror"
		])
}
```

```javascript
import raytracer from 'raytracer-node';

var lastCall = 0;

function createImage(sceneData, sphereData) {
    const emitter = new EventEmitter()

	emitter.on('start', (file) => console.log(file));
	emitter.on('progress', (percent, time) =>
	{
        let splitTime = time.split(" ");
        splitTime[3] = parseFloat(splitTime[3]).toFixed(2);

        let fixedString = splitTime.join(" ");

        if (new Date() - lastCall < 500)
        {
            return false;
        }

        lastCall = new Date();
        console.log("Percent Complete", parseInt(percent.substring(0, 2)), "Time Remaining", fixedString);
    });
    emitter.on('finish', (time) => console.log(time));
    const imageString = raytracer(emitter.emit.bind(emitter), sceneData, sphereData);

    emitter.removeListener('start', () => {});
	emitter.removeListener('progress', () => {});
	emitter.removeListener('finish', () => {});

    return imageString;
}
```

```javascript
function generateHTMLElement(imageString) {
    let imageDataLines = outputImageString.split("\n");
    let imageDataLinesBroken = [];

    imageDataLines.forEach((line) =>
    {
        imageDataLinesBroken.push(line.split(" "));
    });

    let width = imageDataLinesBroken[1][0];
    let height = imageDataLinesBroken[1][1];

    let c = document.createElement("canvas");
    let ctx = c.getContext("2d");

    c.width = width;
    c.height = height;

    let myImageData = ctx.createImageData(width, height);
    let counter = 0;

    let imageData = myImageData.data;

    for (let i = 2; i < imageDataLinesBroken.length; i++)
    {
        for (let j = 0; j < imageDataLinesBroken[i].length; j++)
        {
            imageData[counter] = parseInt(imageDataLinesBroken[i][j]);
            counter++;

            if ((j + 1) % 3 === 0)
            {
                imageData[counter] = 255;
                counter++;
            }
        }
    }

    ctx.putImageData(myImageData, 0, 0);
    return c.toDataURL("image/png");
}
```

### Creating Driver Files

For your convenience, example driver files have been provided under the folder titled ExampleDriverFiles. These driver
files are all fully featured to create interesting and dramatic images that showcase the features of the engine.

If you wish to create your own driver files, the following fields are required for each element:

```
eye X Y Z  
look X Y Z  
up X Y Z  
d Distance  
bounds Left Right Bottom Top  
res Width Height  
sphere X Y Z Radius AlbedoRed AlbedoGreen AlbedoBlue  
model RX RY RZ RTheta ScaleFactor TX TY TZ SmoothingTheta FilePath
```

For Spheres and for Model Material Files, optional fields exist. Appending `light`, `mirror`, or `glass` to the end of a sphere line or material description will activate the corresponding property for that section of the image. See reference driver files for examples. Make sure that the bounds and resolution share an aspect ratio. This is important to ensure no artifacts.

For a node install, you directly input the fields above into the function call. The first arguement to the library will be an emitter, followed by the scene data, and then the sphere data. You must also append the number of samples required to the scene data, as shown in the example above.

## Where to Find OBJ Files

Finding obj files is inconsequential. Simply look up ".obj files" on the internet and choose one that fits your liking.
Make sure that you create the proper material file using the format showcases in the example files. Files with Triangle counts of over 10K will result in images that take a significant time to render.
