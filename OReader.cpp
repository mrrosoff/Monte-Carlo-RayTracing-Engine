//
// Created by Max Rosoff on 9/7/2019.
//

#include "OReader.h"

using namespace std;

OReader::OReader(const string &objName) : objPath(objName) {

    int status = readObject();

    if(status != 0)
    {
        throw "Error with Object Creation"s;
    }
}


int OReader::readObject() {

    ifstream objReader("../" + objPath);

    if (!objReader)
    {
        string err = strerror(errno);
        cerr << "Failure to open Obj File - " << objPath <<  ": " << err <<'\n';
        return 1;
    }

    string objLine;

    while(getline(objReader, objLine))
    {
        lines.push_back(objLine);
    }

    return 0;
}