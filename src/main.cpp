
#include "texture.hpp"
#include "utils/chrono.hpp"
#include "renderingGPU/hello_cuda.hpp"

namespace RT
{	
	int main( int argc, char ** argv )
	{
		const double aspect_ratio = 16.0 / 9.0;
		int nbSample = 32;
		int width = 1920;
		int nbImage = 10;
		int skipImage = 0;
		for (int i = 1; i < argc; i++)
		{
			std::string arg = argv[i];

			if (arg == "-spp" && i + 1 < argc)
			{
				nbSample = std::stoi(argv[++i]);
			}
			else if (arg == "-w" && i + 1 < argc)
			{
				width = std::stoi(argv[++i]);
			}
			else if (arg == "-i" && i + 1 < argc)
			{
				nbImage = std::stoi(argv[++i]);
			}
			else if (arg == "-skip" && i + 1 < argc)
			{
				skipImage = std::stoi(argv[++i]);
			}
			else if (arg == "--help")
			{
				std::cout << "Usage: ./mon_projet [options]\n";
				std::cout << "  -spp <int>   samples per pixel\n";
				std::cout << "  -w <int>     width\n";
				std::cout << "  -i <int>     number of images\n";
				std::cout << "  -skip <int>  start with the ith image\n";
				return 0;
			}
		}

		int			 temp_height  = int( width / aspect_ratio );
		const int	 height	  = ( temp_height < 1 ) ? 1 : temp_height;

		Texture imgCuda =  Texture(width, height);
		float maxElevation = 90.0f;
		Chrono			   chrono;
		chrono.start();
		/*for(int i = skipImage; i < nbImage; i++){
			float t = i / float(nbImage - 1);
			if(nbImage == 1){
				t = 0.5f;
			}
			
			float theta = 1.1 * PIf * t; 
			float az = 20.f * PIf / 180.f;

				Vec3f base = Vec3f(
					cos(theta),
					sin(theta),
					0.0f
				);

				Vec3f sunDir = normalize(Vec3f(
					base.x * cos(az) - base.z * sin(az),
					base.y,
					base.x * sin(az) + base.z * cos(az)
				));
			unsigned char* img_cuda_raw = launchHelloCUDA(nbSample, width, height, sunDir.x, sunDir.y, sunDir.z);
			imgCuda.createFromRaw(img_cuda_raw, width, height);
			const std::string imgCudaName = "imageCuda"+std::to_string(i)+".jpg";
			imgCuda.saveJPG(RESULTS_PATH + imgCudaName);
			std::cout << "saved" +std::to_string(i)<< std::endl;
		}*/
		float t = 0.5f;
		float theta = 1.1 * PIf * t;
		float az = 20.f * PIf / 180.f;
		Vec3f base = Vec3f(
			cos(theta),
			sin(theta),
			0.0f
		);
		Vec3f sunDir = normalize(Vec3f(
			base.x * cos(az) - base.z * sin(az),
			base.y,
			base.x * sin(az) + base.z * cos(az)
		));
		unsigned char* img_cuda_raw = launchHelloCUDA(nbSample, width, height, sunDir.x, sunDir.y, sunDir.z);
		imgCuda.createFromRaw(img_cuda_raw, width, height);
		const std::string imgCudaName = "imageCuda.jpg";
		imgCuda.saveJPG(RESULTS_PATH + imgCudaName);
		std::cout << "saved" << std::endl;

		chrono.stop();
		float time = chrono.elapsedTime();
		int minutes = (int)time / 60.f;
		int seconds = (int)time % 60;
		std::cout << "Done in " << time << "s (" << minutes << "m and " << seconds << "s)" << std::endl;


		return EXIT_SUCCESS;
	}
} // namespace RT

int main( int argc, char ** argv )
{
	try
	{
		return RT::main( argc, argv );
	}
	catch ( const std::exception & e )
	{
		std::cerr << "Exception caught:" << std::endl << e.what() << std::endl;
	}
}
