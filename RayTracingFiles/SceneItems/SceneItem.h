//
// Created by Max Rosoff on 11/17/2019.
//

#ifndef RAYTRACER_SCENEITEM_H
#define RAYTRACER_SCENEITEM_H

#include <exception>

#include "../SceneItems/Material.h"
#include "../RayTracing/Ray.h"

#include "../../Eigen/Eigen"

class SceneItem {

public:

    virtual bool intersectionTest(Ray &) const = 0;
    virtual Ray makeExitRefrationRay(const Ray &, double, double) const = 0;
    Eigen::Vector3d doSnellsLaw(const Ray &, double, double) const;
    
    bool isLight = false;
};

#endif //RAYTRACER_SCENEITEM_H
