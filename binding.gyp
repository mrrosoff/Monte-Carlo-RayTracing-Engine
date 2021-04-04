{
  "targets": [
    {
      "target_name": "raytracer",
      "cflags_cc": [ "-std=c++14", "-Wall", "-fopenmp", "-fexceptions", "-DNODE"],
      'conditions': [
        ['OS=="mac"', {
            'xcode_settings': {
                'GCC_ENABLE_CPP_EXCEPTIONS': 'YES'
            }
        }]
      ],
      "sources": [
        "./RayTracingFiles/index.cpp",
        "./RayTracingFiles/FileInOut/DReader.cpp",
        "./RayTracingFiles/Matrix/Matrix.cpp",
        "./RayTracingFiles/Matrix/Vector.cpp",
        "./RayTracingFiles/RayTracing/Camera.cpp",
        "./RayTracingFiles/RayTracing/Ray.cpp",
        "./RayTracingFiles/RayTracing/RayTracer.cpp",
        "./RayTracingFiles/SceneItems/Models/Face.cpp",
        "./RayTracingFiles/SceneItems/Models/Object.cpp",
        "./RayTracingFiles/SceneItems/Models/Remap.cpp",
        "./RayTracingFiles/SceneItems/Models/Vertex.cpp",
        "./RayTracingFiles/SceneItems/Material.cpp",
        "./RayTracingFiles/SceneItems/SceneItem.cpp",
        "./RayTracingFiles/SceneItems/Spheres/Sphere.cpp",
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ],
    }
  ]
}
