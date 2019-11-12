//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_LIGHTSOURCE_H
#define GRAPHICS_LIGHTSOURCE_H

#include <iostream>

#include "Eigen"

class LightSource {

public:

    LightSource() = delete;
    LightSource(const LightSource &) = default;
    LightSource &operator=(const LightSource &) = delete;
    ~LightSource() = default;

    explicit LightSource(const Eigen::Vector3d &, const Eigen::Vector3d &, double);

    Eigen::Vector3d position;
    Eigen::Vector3d rgb;
    double w = 0;

};

std::ostream &operator<<(std::ostream &out, const LightSource &light);


#endif //GRAPHICS_LIGHTSOURCE_H
