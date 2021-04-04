#include <string>
#include <vector>
#include <iostream>

#include <napi.h>

#include "RayTracing/RayTracer.h"

using namespace std;

string doMainRun(const Napi::Env &env, const Napi::Function &emit, const vector<string> &sceneData, const vector<string> &spheres)
{
    return RayTracer().rayTrace(env, emit, sceneData, spheres);
}

Napi::String raytrace(const Napi::CallbackInfo &info) {

    Napi::Env env = info.Env();

    Napi::Function emit = info[0].As<Napi::Function>();

    Napi::Array sceneData = info[1].As<Napi::Array>();
    Napi::Array spheres = info[2].As<Napi::Array>();

    vector<string> sceneDataVector;

    for (size_t i = 0; i < sceneData.Length(); i++)
    {
        Napi::Value value = sceneData[i];
        sceneDataVector.push_back(value.ToString());
    }

    vector<string> spheresVec;

    for (size_t i = 0; i < spheres.Length(); i++)
    {
        Napi::Value value = spheres[i];
        spheresVec.push_back(value.ToString());
    }

    string result = doMainRun(env, emit, sceneDataVector, spheresVec);

    return Napi::String::New(env, result);
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {

    exports.Set(Napi::String::New(env, "raytrace"), Napi::Function::New(env, raytrace));

    return exports;
}


NODE_API_MODULE(raytracer, Init)
