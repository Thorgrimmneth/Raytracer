#include "application.hpp"
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include "utils/png_saver.hpp"

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
        else if (arg == "-rng" && i + 1 < argc)
        {
            rngManip = std::stoi(argv[++i]);
        }
        else if (arg == "-saveData")
        {
            saveData = true;
        }
        else if (arg == "-printValue")
        {
            printValue = true;
        }
        else if (arg == "-output")
        {
            output_image = true;
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
    float3 cam_pos = make_float3(0.f, 3.f, 8.f);
    float3 cam_target = make_float3(0.f, 1.f, 0.f);
    float3 cam_up = make_float3(0.f, 1.f, 0.f);
    float3 forward = normalize(cam_target - cam_pos);
    float3 right = normalize(cross(forward, cam_up));
    float3 up = normalize(cross(right, forward));

    float theta = PIf * t;
    float3 sun_direction = normalize(-forward * cos(theta) + up * sin(theta));
    return sun_direction;
}

int Application::launchApp(int argc, char **argv)
{

    int result = initParameters(argc, argv);
    if (result == 1)
        return 1;

    Chrono chronoGlobal;
    chronoGlobal.start();
    Chrono chrono;
    float value = 1000.f;
    if (saveData)
    {
        std::string result_numbers = "nbRPP = " + std::to_string(nbRPP) + "; width = " + std::to_string(width) +
                                     "; height = " + std::to_string(height) + "; t = " + std::to_string(t) +
                                     "; convergence = " + std::to_string(convergence) +
                                     "; threshold = " + std::to_string(threshold) + "\n";
        result_numbers += "kernel name; kernel time (ms)";
        std::ofstream csvFile(RESULTS_PATH + "../results.csv", std::ios::out | std::ios::trunc);
        csvFile << result_numbers << std::endl;
        csvFile.close();
    }
    // performance mode, no GUI. Used for profiling and creating final images
    if (mode == 0)
    {
        chrono.start();
        sunDir = computeSunDir(t);

        Renderer renderer;
        renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z);
        for (int i = 0; i < nbRPP && value > threshold; i++)
        {
            std::string result_numbers = "";
            value = renderer.render(false, convergence, saveData, &result_numbers);
            if (saveData)
            {
                std::ofstream csvFile(RESULTS_PATH + "../results.csv", std::ios::app);
                csvFile << result_numbers << std::endl;
                csvFile.close();
            }
            if (printValue && i % 1000 == 0)
            {
                std::cout << "image : " << i << " samples, error = " << value << std::endl;
            }
        }
        if (output_image)
        {
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
                std::cout << "converged after " << renderer.get_frame_number() << " with " << value << " error"
                          << std::endl;
                const std::string imageName = "performance.png";
                writePNG((char*)(RESULTS_PATH + imageName).c_str(), img_data, width, height);
                std::cout << "saved : " + imageName << std::endl;
                chrono.stop();
                std::cout << "avg : " << renderer.get_frame_number() / (chrono.elapsedTime()) << " spp/s" << std::endl;
                free(img_data);
                free(d_finalizedImage);
            }
        }
    }
    // cumulative mode. GUI, fps count
    else if (mode == 1)
    {
        // setup window for cumulative rendering
        Window win(width, height);

        sunDir = computeSunDir(t);

        unsigned char *img_cuda_raw = win.cumulativeRendering(sunDir, width, height, convergence, threshold, rngManip);
        
        // end of rendering
            writePNG((char*)(RESULTS_PATH + "cumulative.png").c_str(), img_cuda_raw, width, height);
            const std::string imageName = "cumulative.png";
            std::cout << "saved : " + imageName << std::endl;
    }
    else if (mode == 2)
    {
        chrono.start();
        sunDir = computeSunDir(t);

        Renderer renderer;
        renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z);
        while (value > threshold)
        {
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
            std::cout << "converged after " << renderer.get_frame_number() << " with " << value << " error"
                      << std::endl;
            const std::string imageName = "performance.png";
            writePNG((char*)(RESULTS_PATH + imageName).c_str(), img_data, width, height);
            std::cout << "saved : " + imageName << std::endl;
            chrono.stop();
            std::cout << "avg : " << renderer.get_frame_number() / (chrono.elapsedTime()) << " spp/s" << std::endl;
            free(img_data);
            free(d_finalizedImage);
        }
    }
    else if (mode >= 3)
    {
        for (int i = 0; i < nbImage; i++)
        {
            value = 1000.f;
            chrono.start();
            t = (float)i / (float)nbImage;
            sunDir = computeSunDir(t);

            Renderer renderer;
            renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z, i);
            printf("Rendering image %d/%d\n", i + 1, nbImage);
            for (int j = 0; j < nbRPP && value > threshold; j++)
            {
                value = renderer.render(false, convergence);
                if (j % 1000 == 0)
                {
                    std::cout << "image " << i << " : " << j << " samples, error = " << value << std::endl;
                }
            }

            // Get finalized image from GPU and save to texture
            float3 *d_finalizedImage = renderer.get_finalized_image();
            if (d_finalizedImage)
            {
                chrono.stop();
                printf("image %d : converged after %d samples with %f error in %fs (around %f spp/s)\n", i,
                       renderer.get_frame_number(), value, chrono.elapsedTime(),
                       renderer.get_frame_number() / chrono.elapsedTime());
                unsigned char *img_data = (unsigned char *)malloc(width * height * 3);
                for (int k = 0; k < width * height; k++)
                {
                    img_data[k * 3] = static_cast<unsigned char>(d_finalizedImage[k].x * 255.0f);
                    img_data[k * 3 + 1] = static_cast<unsigned char>(d_finalizedImage[k].y * 255.0f);
                    img_data[k * 3 + 2] = static_cast<unsigned char>(d_finalizedImage[k].z * 255.0f);
                }
                const std::string imageName = "performance_" + std::to_string(i) + ".png";
                writePNG((char*)(RESULTS_PATH + imageName).c_str(), img_data, width, height);
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