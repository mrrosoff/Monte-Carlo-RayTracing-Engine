//
// Created by Max Rosoff on 9/7/2019.
//


#include "DReader.h"

using namespace std;

DReader::DReader(const string &driverFile) : driverPath(driverFile), driverName(findDriverName(driverFile)) {

    int status = readDriver();

    if(status != 0)
    {
        throw "Error with Driver Creation"s;
    }
}


string DReader::findDriverName(const string &driverFile) {

    unsigned long pIndex = driverFile.find_last_of('.');
    unsigned long sIndex = driverFile.find_last_of('/');
    sIndex += 1;
    return driverFile.substr(sIndex, pIndex - sIndex);

}


int DReader::readDriver() {

    ifstream driverReader(driverPath);

    if (!driverReader) {
        string err = strerror(errno);
        cerr << "Failure to open Driver File - " << driverPath << ": " << err << '\n';
        return 1;
    }

    string driverLine;

    while(getline(driverReader, driverLine))
    {
        stringstream lineReader(driverLine);
        string token;

        vector<string> lineData;

        while(getline(lineReader, token, ' '))
        {
            lineData.push_back(token);
        }

        remaps.emplace_back(lineData);
    }

    return 0;
}
