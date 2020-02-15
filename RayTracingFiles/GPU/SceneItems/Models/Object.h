//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_OBJECT_H
#define GRAPHICS_OBJECT_H

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#include <cerrno>
#include <cstring>

#include "../../RayTracing/Ray.h"
#include "../SceneItem.h"
#include "../Material.h"

#include "./Remap.h"
#include "./Vertex.h"
#include "./Face.h"

class Object : public SceneItem {

public:

    Object() = default;
    Object(const Object &) = default;
    Object &operator=(const Object &) = default;
    virtual ~Object()= default;

    __host__ explicit Object(const Remap &);

    __device__ bool intersectionTest(Ray &ray) const override;
    __device__ Ray makeExitRefrationRay(const Ray &, double, double) const override;

    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::string objPath;
    double smoothingAngle = 0;

private:

    __host__ void readObject(const Remap &);
    __host__ void readObjectFaceLine(const std::vector<std::string> &, int);
    __host__ void readMaterialFile(const std::string &);
    __host__ void calculateNormals();
};

__host__ std::ostream &operator<<(std::ostream &, const Object &);


#endif //GRAPHICS_OBJECT_H
