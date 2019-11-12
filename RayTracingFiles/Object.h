//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_OBJECT_H
#define GRAPHICS_OBJECT_H

#include <string>
#include <vector>
#include <tuple>

#include <iostream>
#include <fstream>
#include <sstream>

#include <cerrno>
#include <cstring>

#include "Ray.h"
#include "Remap.h"

class Object {

public:

    Object() = default;
    Object(const Object &) = default;
    Object &operator=(const Object &) = default;
    virtual ~Object()= default;

    explicit Object(const Remap &);
    bool intersectionTest(Ray &ray) const;

    std::vector<Material> materials;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::tuple<std::vector<std::vector<int>>, int>> faces;

private:

    void readObject(const Remap &);
    void readMaterialFile(std::vector<std::string> &);

    std::string objPath;
};

std::ostream &operator<<(std::ostream &, const Object &);


#endif //GRAPHICS_OBJECT_H
