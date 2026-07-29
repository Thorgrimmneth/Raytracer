#include "utils/texture.hpp"
#include "utils/chrono.hpp"

#include "window.hpp"

class Application {
  public:
    Application() = default;
    ~Application() = default;

    /*Application(int argc, char **argv)
    {
        initParameters(argc, argv);
    }*/

    const double aspect_ratio = 16.0 / 9.0;
    int width = 1920;
    int height = 1080;
    int nbRPP = 1;
    int nbImage = 10;
    int skipImage = 0;
    int rngManip = 0;
    float t = 0.5f;
    int mode = 0;
    float threshold = 0.001f;
    bool output_image = false;
    bool convergence = false;
    bool saveData = false;
    bool printValue = false;
    float maxElevation = 90.0f;
    float3 sunDir = make_float3(0.f, 1.f, 0.f);

    int launchApp(int argc, char **argv);

  private:
    int initParameters(int argc, char **argv);
    float3 computeSunDir(float t);
};
