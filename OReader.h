//
// Created by Max Rosoff on 9/7/2019.
//

#ifndef GRAPHICS_OREADER_H
#define GRAPHICS_OREADER_H

#include <string>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>

#include <cerrno>
#include <cstring>

class OReader {

public:

    explicit OReader(const std::string &);

    OReader(const OReader &) = delete;
    OReader &operator=(const OReader &) = delete;
    ~OReader()= default;

    std::vector<std::string> lines;

private:

    int readObject();

    std::string objPath;

};


#endif //GRAPHICS_OREADER_H
