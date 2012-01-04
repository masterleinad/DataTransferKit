//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh/Mesh_BoundingBox.hpp
 * \author Stuart Slattery
 * \date   Thu Dec 08 09:58:46 2011
 * \brief  Bounding box class definition.
 */
//---------------------------------------------------------------------------//

#ifndef mesh_Mesh_BoundingBox_hpp
#define mesh_Mesh_BoundingBox_hpp

#include <Mesh_Point.hpp>

#include "Teuchos_Tuple.hpp"

namespace Coupler
{

//===========================================================================//
/*!
 * \class Mesh_BoundingBox
 * \brief A bounding box definition for local meshes.
 *
 * A class for encapsulating the local mesh bounding box.
 */
/*! 
 * \example mesh/test/tstMesh_BoundingBox.cpp
 *
 * Test of BoundingBox.
 */
//===========================================================================//

template<class HandleType_T, class CoordinateType_T>
class BoundingBox 
{
  public:

    //@{ 
    //! Useful typedefs.
    typedef HandleType_T                               HandleType;
    typedef CoordinateType_T                           CoordinateType;
    typedef Point<HandleType,CoordinateType>           PointType;
    typedef Teuchos::Tuple<CoordinateType,6>           Domain;
    //@}

  private:

    // Box domain (imin,imax,jmin,jmax,kmin,kmax)
    Domain d_domain;

  public:

    //! Constructor.
    BoundingBox(CoordinateType i_min,
		 CoordinateType i_max,
		 CoordinateType j_min,
		 CoordinateType j_max,
		 CoordinateType k_min,
		 CoordinateType k_max)
    {
	d_domain[0] = i_min;
	d_domain[1] = i_max;
	d_domain[2] = j_min;
	d_domain[3] = j_max;
	d_domain[4] = k_min;
	d_domain[5] = k_max;
    }

    //! Copy constructor.
    template<class HandleType_2, class CoordinateType_2>
    BoundingBox(const BoundingBox<HandleType_2,CoordinateType_2>& bounding_box)
	: d_domain( bounding_box.d_domain )
    { /* ... */}

    //@{
    //! Get the domain.
    Domain  domain() const { return d_domain; }
    Domain& domain()       { return d_domain; }
    //@}

    // Query the bounding box with a point.
    inline bool point_query(PointType point);
};

} // end namespace Coupler

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//

#include "Mesh_BoundingBox.i.hpp"

#endif // mesh_Mesh_BoundingBox_hpp

//---------------------------------------------------------------------------//
//              end of mesh/Mesh_BoundingBox.hpp
//---------------------------------------------------------------------------//
