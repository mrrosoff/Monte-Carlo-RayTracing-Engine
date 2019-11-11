//
// Created by Max Rosoff on 9/21/2019.
//

#ifndef GRAPHICS_PWRITER_H
#define GRAPHICS_PWRITER_H

#include <string>
#include <vector>

#include <fstream>

#include <cerrno>
#include <cstring>

class PWriter {

public:

    explicit PWriter(const std::string &);
    PWriter &operator<<(const std::vector<std::vector<std::vector<int>>> &);

    PWriter(const PWriter &) = delete;
    PWriter &operator=(const PWriter &) = delete;
    ~PWriter() = default;

private:

    std::string outFile;
};


#endif //GRAPHICS_PWRITER_H
