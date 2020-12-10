//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_OBJECT_H
#define GRAPHICS_OBJECT_H

#define _USE_MATH_DEFINES
#include <cmath>

#include <string>
#include <vector>
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

    explicit Object(const Remap &);

    bool intersectionTest(Ray &ray) const override;
    Ray makeExitRefrationRay(const Ray &, double, double) const override;

    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::string objPath;
    double smoothingAngle = 0;

private:

    void readObject(const Remap &);
    void readObjectFaceLine(const std::vector<std::string> &, int);
    void readMaterialFile(const std::string &);
    void calculateNormals();
};

std::ostream &operator<<(std::ostream &, const Object &);


#endif //GRAPHICS_OBJECT_H
