#include "renderer.hpp"
#include "cameras/perspective_camera.hpp"
#include "integrators/direct_lighting_integrator.hpp"
#include "integrators/ray_cast_integrator.hpp"
#include "integrators/whitted_integrator.hpp"
#include "utils/console_progress_bar.hpp"

namespace RT
{
	Renderer::Renderer() { _integrator = new RayCastIntegrator(); }

	void Renderer::setIntegrator( const IntegratorType p_integratorType)
	{
		if ( _integrator != nullptr ) { delete _integrator; }

		switch ( p_integratorType )
		{
		case IntegratorType::DIRECT_LIGHTING:
		{
			_integrator = new DirectLightingIntegrator();
			break;
		}
		case IntegratorType::RAY_CAST:
		{
			_integrator = new RayCastIntegrator();
			break;
		}
		case IntegratorType::WHITTED_LIGHTING:
		{
			_integrator = new WhittedIntegrator();
			break;
		}
		default:
		{
			_integrator = new RayCastIntegrator();
			break;
		}
		}
	}

	void Renderer::setBackgroundColor( const Vec3f & p_color )
	{
		if ( _integrator == nullptr ) { std::cout << "[Renderer::setBackgroundColor] Integrator is null" << std::endl; }
		else { _integrator->setBackgroundColor( p_color ); }
	}

	float Renderer::renderImage( const Scene & p_scene, const BaseCamera * p_camera, Texture & p_texture )
	{
		const int width	 = p_texture.getWidth();
		const int height = p_texture.getHeight();

		Chrono			   chrono;
		ConsoleProgressBar progressBar;

		progressBar.start( height, 50 );
		chrono.start();

		#pragma omp parallel for
		for ( int j = 0; j < height; j++ )
		{
			for ( int i = 0; i < width; i++ )
			{
				Vec3f luminance = Vec3f( 0.f );
				for ( int rayNumber = 0; rayNumber < _nbPixelSamples; rayNumber++ )
				{
					float random_i		 = randomFloat();
					float random_j		 = randomFloat();
					float interpolized_i = (float)( i + random_i ) / (float)( width - 1 );
					float interpolized_j = (float)( j + random_j ) / (float)( height - 1 );
					Ray	  ray			 = p_camera->generateRay( interpolized_i, interpolized_j );
					luminance += _integrator->Li( p_scene, ray, 0.f, 10000000.f );
				}
				luminance = glm::clamp( luminance / (float)_nbPixelSamples, 0.f, 1.f );

				p_texture.setPixel( i, j, luminance );
			}
			progressBar.next();
		}

		chrono.stop();
		progressBar.stop();

		return chrono.elapsedTime();
	}
} // namespace RT
