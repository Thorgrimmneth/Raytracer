#include "application.hpp"
#include <nvtx3/nvToolsExt.h>

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
        else if (arg == "-t" && i + 1 < argc)
        {
            t = std::stof(argv[++i]);
        }
        else if (arg == "-convergence" && i + 1 < argc)
        {
            convergence = std::stof(argv[++i]) != 0;
        }
        else if (arg == "-threshold" && i + 1 < argc)
        {
            threshold = std::stof(argv[++i]);
            convergence = true;
        }
        else if(arg == "-rng" && i + 1 < argc)
        {
            rngManip = std::stoi(argv[++i]);
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: ./mon_projet [options]\n";
            std::cout << "  -rpp <int>   samples per pixel\n";
            std::cout << "  -w <int>     width\n";
            std::cout << "  -i <int>     number of images\n";
            std::cout << "  -skip <int>  start with the ith image\n";
            std::cout << "  -mode        0 = profiling mode, 1 = cumulative mode, 2 = performance mode\n";
            std::cout << "  -t <float>   time parameter\n";
            std::cout << "  -convergence <int> runs until convergence is reached\n";
            std::cout << "  -threshold <float> threshold for the convergence\n";
            std::cout << "  -rng <int>   change rng\n";
            return 1;
        }
    }

    // scale height on width
    height = int(width / aspect_ratio);
    return 0;
}

float3 Application::computeSunDir(float t)
{
    float theta = 1.1 * PIf * t;
    float az = 20.f * PIf / 180.f;
    float3 base = make_float3(cos(theta), sin(theta), 0.0f);
    float3 sun_direction = normalize(make_float3(base.x * cos(az) - base.z * sin(az), base.y, base.x * sin(az) + base.z * cos(az)));
    return sun_direction;
}

int Application::launchApp(int argc, char **argv)
{

    int result = initParameters(argc, argv);
    if (result == 1)
        return 1;

    image = Texture(width, height);

    Chrono chronoGlobal;
    chronoGlobal.start();
    Chrono chrono;
    float value = 1000.f;
    // performance mode, no GUI. Used for profiling and creating final images
    if (mode == 0)
    {
        sunDir = computeSunDir(t);

        Renderer renderer;
        renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z);
        for (int i = 0; i < nbRPP && value > threshold; i++)
        {
            value = renderer.render(false, convergence);
            value = 0.f;
        }
    }
    // cumulative mode. GUI, fps count. Allows to switch between megakernel and wavefront
    else if (mode == 1)
    {
        // setup window for cumulative rendering
        Window win(width, height);

        sunDir = computeSunDir(t);

        unsigned char *img_cuda_raw = win.cumulativeRendering(sunDir, width, height, convergence, threshold, rngManip);
        
        // end of rendering
        image.createFromRaw(img_cuda_raw, width, height);
        const std::string imageName = "cumulative.jpg";
        image.saveJPG(RESULTS_PATH + imageName);
        std::cout << "saved : " + imageName << std::endl;
    }
    else if (mode == 2)
    {
        chrono.start();
        sunDir = computeSunDir(t);

        Renderer renderer;
        renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z);
        while (value > threshold){
            value = renderer.render(false, true);
        }

        // Get finalized image from GPU and save to texture
        float3 *d_finalizedImage = renderer.get_finalized_image();
        if (d_finalizedImage)
        {
            unsigned char *img_data = (unsigned char *)malloc(width * height * 3);
            
            for (int i = 0; i < width * height; i++)
            {
                img_data[i * 3] = static_cast<unsigned char>(d_finalizedImage[i].x * 255.0f);
                img_data[i * 3 + 1] = static_cast<unsigned char>(d_finalizedImage[i].y * 255.0f);
                img_data[i * 3 + 2] = static_cast<unsigned char>(d_finalizedImage[i].z * 255.0f);
            }
            std::cout << "converged after " << renderer.get_frame_number() << " with " << value << " error" << std::endl;
            image.createFromRaw(img_data, width, height);
            const std::string imageName = "performance.jpg";
            image.saveJPG(RESULTS_PATH + imageName);
            std::cout << "saved : " + imageName << std::endl;
            chrono.stop();
            std::cout << "avg : " << renderer.get_frame_number() / (chrono.elapsedTime()) << " spp/s" << std::endl;
            free(img_data);
            free(d_finalizedImage);
        }
    }
    else if(mode >=3)
    {
        for(int i = 0; i < nbImage; i++)
        {
            value = 1000.f;
            chrono.start();
            sunDir = computeSunDir(t);

            Renderer renderer;
            renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z, i);
            printf("Rendering image %d/%d\n", i + 1, nbImage);
            for (int j = 0; j < nbRPP && value > threshold; j++)
            {
                value = renderer.render(false, convergence);
                if(j % 1000 == 0)
                {
                    std::cout << "image " << i << " : " << j << " samples, error = " << value << std::endl;
                }
            }

            // Get finalized image from GPU and save to texture
            float3 *d_finalizedImage = renderer.get_finalized_image();
            if (d_finalizedImage)
            {
                chrono.stop();
                printf("image %d : converged after %d samples with %f error in %fs (around %f spp/s)\n", i, renderer.get_frame_number(), value, chrono.elapsedTime(), renderer.get_frame_number() / chrono.elapsedTime());
                unsigned char *img_data = (unsigned char *)malloc(width * height * 3);
                for (int k = 0; k < width * height; k++)
                {
                    img_data[k * 3] = static_cast<unsigned char>(d_finalizedImage[k].x * 255.0f);
                    img_data[k * 3 + 1] = static_cast<unsigned char>(d_finalizedImage[k].y * 255.0f);
                    img_data[k * 3 + 2] = static_cast<unsigned char>(d_finalizedImage[k].z * 255.0f);
                }
                image.createFromRaw(img_data, width, height);
                const std::string imageName = "performance_" + std::to_string(i) + ".jpg";
                image.saveJPG(RESULTS_PATH + imageName);
                std::cout << "saved : " + imageName << std::endl;
                free(img_data);
                free(d_finalizedImage);
            }
        }
    }

    chronoGlobal.stop();
    float time = chronoGlobal.elapsedTime();
    int minutes = (int)time / 60.f;
    int seconds = (int)time % 60;
    std::cout << "Done in " << time << "s (" << minutes << "m and " << seconds << "s)" << std::endl;

    return EXIT_SUCCESS;
}