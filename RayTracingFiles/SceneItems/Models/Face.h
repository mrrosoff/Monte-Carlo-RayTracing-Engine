//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_FACE_H
#define GRAPHICS_FACE_H

#include <vector>
#include <iostream>

#include "../../../Eigen/Eigen"

class Face {

public:

    Face() = default;
    Face(const Face &) = default;
    Face &operator=(const Face &) = default;
    virtual ~Face()= default;

    explicit Face(const std::vector<int> &, int);

    std::vector<int> vertexIndexList;
    int materialIndex;
    std::vector<Eigen::Vector3d> normals;
};

std::ostream &operator<<(std::ostream &, const Face &);


#endif //GRAPHICS_FACE_H
