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

#include "../Matrix/Matrix.h"
#include "../RayTracing/Camera.h"
#include "../SceneItems/Models/Object.h"
#include "../SceneItems/Material.h"
#include "../SceneItems/SceneItem.h"
#include "../SceneItems/Sphere.h"
#include "../SceneItems/Models/Remap.h"

class DReader {

public:

    DReader() = default;
    DReader(const DReader &) = default;
    DReader &operator=(const DReader &) = delete;
    ~DReader() = default;

    __host__ explicit DReader(const std::string &);

    std::string driverName;
    std::string driverFile;

    Camera camera;
    std::vector<std::shared_ptr<SceneItem>> items;

private:

    __host__ void readDriver(const std::string &);
    __host__ static void throwErrorMessage(int, int, const std::string &);

    __host__ std::string findDriverName(const std::string &);
    __host__ Vector<3> parseEye(const std::vector<std::string> &) const;
    __host__ Vector<3> parseLook(const std::vector<std::string> &) const;
    __host__ Vector<3> parseUp(const std::vector<std::string> &) const;
    __host__ double parseD(const std::vector<std::string> &) const;
    __host__ Vector<4> parseBounds(const std::vector<std::string> &) const;
    __host__ Vector<2> parseRes(const std::vector<std::string> &) const;
    __host__ void parseSphere(const std::vector<std::string> &);
    __host__ void parseModel(const std::vector<std::string> &);
};


#endif //GRAPHICS_DRIVERREADER_H
