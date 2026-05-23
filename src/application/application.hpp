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
    int nbRPP = 32;
    int nbImage = 10;
    int skipImage = 0;
    float t = 0.5f;
    int mode = 0;
    float threshold = 0.001f;
    bool convergence = false;
    float maxElevation = 90.0f;
    Vec3f sunDir = Vec3f(0.f, 1.f, 0.f);
    Texture image;

    int launchApp(int argc, char **argv);

  private:
    int initParameters(int argc, char **argv);
    Vec3f computeSunDir(float t);
};
