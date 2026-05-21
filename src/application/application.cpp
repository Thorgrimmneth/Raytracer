#include "application.hpp"

int Application::initParameters(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-rpp" && i + 1 < argc)
        {
            nbRPP = std::stoi(argv[++i]);
        }
        else if (arg == "-w" && i + 1 < argc)
        {
            width = fmax(144, std::stoi(argv[++i]));
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            nbImage = std::stoi(argv[++i]);
        }
        else if (arg == "-skip" && i + 1 < argc)
        {
            skipImage = std::stoi(argv[++i]);
        }
        else if (arg == "-mode" && i + 1 < argc)
        {
            mode = std::stoi(argv[++i]);
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: ./mon_projet [options]\n";
            std::cout << "  -spp <int>   samples per pixel\n";
            std::cout << "  -w <int>     width\n";
            std::cout << "  -i <int>     number of images\n";
            std::cout << "  -skip <int>  start with the ith image\n";
            std::cout << "  -mode        0 = performance mode, 1 = cumulative mode\n";
            return 1;
        }
    }

    // scale height on width
    height = int(width / aspect_ratio);
    return 0;
}

Vec3f Application::computeSunDir(float t)
{
    float theta = 1.1 * PIf * t;
    float az = 20.f * PIf / 180.f;
    Vec3f base = Vec3f(cos(theta), sin(theta), 0.0f);
    Vec3f sunDirection = normalize(Vec3f(base.x * cos(az) - base.z * sin(az), base.y, base.x * sin(az) + base.z * cos(az)));
    return sunDirection;
}

int Application::launchApp(int argc, char **argv)
{

    int result = initParameters(argc, argv);
    if (result == 1)
        return 1;

    image = Texture(width, height);

    Chrono chrono;
    chrono.start();

    // performance mode, no GUI. Used for profiling and creating final images
    if (mode == 0)
    {
        float t = 0.5f;
        sunDir = computeSunDir(t);

        Renderer renderer;
        renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z);
        for (int i = 0; i < nbRPP; i++)
        {
            renderer.render(false);
        }
    }
    // cumulative mode. GUI, fps count. Allows to switch between megakernel and wavefront
    else if (mode >= 1)
    {
        // setup window for cumulative rendering
        Window win(width, height);

        float t = 0.5f;
        sunDir = computeSunDir(t);

        unsigned char *img_cuda_raw = win.cumulativeRendering(sunDir, width, height);

        // end of rendering
        image.createFromRaw(img_cuda_raw, width, height);
        const std::string imageName = "profiling.jpg";
        image.saveJPG(RESULTS_PATH + imageName);
        std::cout << "saved" << std::endl;
    }

    chrono.stop();
    float time = chrono.elapsedTime();
    int minutes = (int)time / 60.f;
    int seconds = (int)time % 60;
    std::cout << "Done in " << time << "s (" << minutes << "m and " << seconds << "s)" << std::endl;

    return EXIT_SUCCESS;
}