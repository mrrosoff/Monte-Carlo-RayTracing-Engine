//
// Created by Max Rosoff on 11/17/2019.
//

#ifndef RAYTRACER_SCENEITEM_H
#define RAYTRACER_SCENEITEM_H

#include "../SceneItems/Material.h"
#include "../RayTracing/Ray.h"

class SceneItem {

public:

    virtual bool intersectionTest(Ray &) const = 0;
    virtual Ray makeExitRefrationRay(const Ray &, double, double) const = 0;
};

std::ostream &operator<<(std::ostream &, const SceneItem &);

#endif //RAYTRACER_SCENEITEM_H
