//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_DRIVERREADER_H
#define GRAPHICS_DRIVERREADER_H

#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#include <cerrno>
#include <cstring>
#include <exception>
#include <memory>

#include "../RayTracing/Camera.h"
#include "../SceneItems/LightSource.h"
#include "../SceneItems/Models/Object.h"
#include "../SceneItems/Material.h"
#include "../SceneItems/SceneItem.h"
#include "../SceneItems/Sphere.h"
#include "../SceneItems/Models/Remap.h"

#include "../../Eigen/Eigen"

class DReader {

public:

    DReader() = default;
    DReader(const DReader &) = default;
    DReader &operator=(const DReader &) = delete;
    ~DReader() = default;

    DReader &operator<<(const std::string &);

    std::string driverName;
    std::string driverFile;

    Camera camera;
    Eigen::Vector3d ambientLight;
    std::vector<LightSource> lights;
    std::vector<std::shared_ptr<SceneItem>> items;

private:

    void readDriver(const std::string &);
    static void throwErrorMessage(int, int, const std::string &);

    std::string findDriverName(const std::string &);
    Eigen::Vector3d parseEye(const std::vector<std::string> &) const;
    Eigen::Vector3d parseLook(const std::vector<std::string> &) const;
    Eigen::Vector3d parseUp(const std::vector<std::string> &) const;
    double parseD(const std::vector<std::string> &) const;
    Eigen::Vector4d parseBounds(const std::vector<std::string> &) const;
    Eigen::Vector2d parseRes(const std::vector<std::string> &) const;
    void parseSphere(const std::vector<std::string> &);
    void parseModel(const std::vector<std::string> &);
};

std::ostream &operator<<(std::ostream &, const DReader &);


#endif //GRAPHICS_DRIVERREADER_H
