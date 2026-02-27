#include "cameras/perspective_camera.hpp"
#include "defines.hpp"
#include "renderer.hpp"
#include "renderingGPU/hello_cuda.hpp"

namespace RT
{	
	//extern "C" void launch_cuda_test();
	int main( int argc, char ** argv )
	{
		const double aspect_ratio = 16.0 / 9.0;
		const int img_width	= argc < 2 ? 600 : glm::max(100,std::atoi(argv[1]));
		int			 temp_height  = int( img_width / aspect_ratio );
		const int	 img_height	  = ( temp_height < 1 ) ? 1 : temp_height;
		Texture imgCuda =  Texture(img_width, img_height);
		RT::Scene scene;
		scene.init();
		int nbSample = argc < 3 ? 32 : glm::max(1,std::atoi(argv[2]));
		float elevation = 50.f;
		float azimuth = 20.f;
		unsigned char* img_cuda_raw = launchHelloCUDA(scene, nbSample, img_width, img_height, elevation, azimuth);
		imgCuda.createFromRaw(img_cuda_raw, img_width, img_height);
		const std::string imgCudaName = "imageCuda.jpg";
		imgCuda.saveJPG(RESULTS_PATH + imgCudaName);
		std::cout << "saved" << std::endl;

		/*
		// Create and setup the renderer.
		int nbPixelSamples = argc < 3 ? 600 : glm::max(100,std::atoi(argv[2]));
		Renderer renderer;
		renderer.setIntegrator( IntegratorType::WHITTED_LIGHTING);
		renderer.setBackgroundColor( Vec3f( 22, 67, 117 ) / 255.f );
		renderer.setNbPixelSamples(nbPixelSamples);
		
		// Launch rendering.
		std::cout << "Rendering..." << std::endl;
		std::cout << "- Image size: " << img_width << "x" << img_height << std::endl;
		std::cout << "Number of sample per pixel: " << nbPixelSamples << std::endl;

		float renderingTime = renderer.renderImage( scene, &camera, img );

		std::cout << "-> Done in " << renderingTime << "ms" << std::endl;

		// Save rendered image.
		const std::string imgName = "image.jpg";
		img.saveJPG( RESULTS_PATH + imgName );*/

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
