#include "area_light.hpp"

namespace RT
{
	LightSample AreaLight::sample( const Vec3f & p_point ) const
	{
		const TriangleMeshGeometry & triangle = getRandomTriangle( _mesh->getTriangles() );
		// Get the area of the triangle for the PDF
		float area = triangle.getArea();
		if ( area <= 0.f ) return {};

		// Sample a point on the triangle
		Vec3f pos = triangle.samplePoint();

		Vec3f dir  = glm::normalize( pos - p_point );
		float dist = glm::distance( pos, p_point );

		Vec3f normal   = triangle.getFaceNormal();
		float cosTheta = glm::dot( normal, -dir );

		if ( cosTheta <= 0.f ) return {};

		// Compute the PDF
		cosTheta	   = std::max( cosTheta, 0.f );
		float pdf	   = ( dist * dist ) / ( area * cosTheta );
		pdf			   = std::max( pdf, 1e-4f );
		Vec3f radiance = ( _color * _power ) / pdf;

		LightSample sample;
		sample._radiance  = radiance;
		sample._pdf		  = pdf;
		sample._power	  = _power;
		sample._distance  = dist;
		sample._direction = dir;

		return sample;
	}
} // namespace RT