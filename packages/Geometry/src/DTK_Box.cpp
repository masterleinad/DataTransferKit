//---------------------------------------------------------------------------//
/*
  Copyright (c) 2014, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the Oak Ridge National Laboratory nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
  * \file DTK_Box.cpp
  * \author Stuart R. Slattery
  * \brief Bounding box definition.
  */
//---------------------------------------------------------------------------//

#include "DTK_Box.hpp"
#include "DTK_DBC.hpp"
#include "DTK_DataSerializer.hpp"

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \brief Default constructor.
 */
Box::Box()
    : d_global_id( dtk_invalid_entity_id )
    , d_owner_rank( -1 )
    , d_x_min( 0.0 )
    , d_y_min( 0.0 )
    , d_z_min( 0.0 )
    , d_x_max( 0.0 )
    , d_y_max( 0.0 )
    , d_z_max( 0.0 )
    , d_byte_size( sizeof(EntityId) + sizeof(int) + 6*sizeof(double) )
{ 
    d_centroid[0] = 0.0;
    d_centroid[1] = 0.0;
    d_centroid[2] = 0.0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 *
 * \param x_min Minimum x coordinate value in the box.
 *
 * \param y_min Minimum y coordinate value in the box.
 *
 * \param z_min Minimum z coordinate value in the box.
 *
 * \param x_max Maximum x coordinate value in the box.
 *
 * \param y_max Maximum y coordinate value in the box.
 *
 * \param z_max Maximum z coordinate value in the box.
 */
Box::Box( const EntityId global_id, const int owner_rank,
	  const double x_min, const double y_min, const double z_min,
	  const double x_max, const double y_max, const double z_max )
    : d_global_id( global_id )
    , d_owner_rank( owner_rank )
    , d_x_min( x_min )
    , d_y_min( y_min )
    , d_z_min( z_min )
    , d_x_max( x_max )
    , d_y_max( y_max )
    , d_z_max( z_max )
    , d_byte_size( sizeof(EntityId) + sizeof(int) + 6*sizeof(double) )
{
    DTK_REQUIRE( d_x_min <= d_x_max );
    DTK_REQUIRE( d_y_min <= d_y_max );
    DTK_REQUIRE( d_z_min <= d_z_max );
    d_centroid[0] = (d_x_max + d_x_min) / 2.0;
    d_centroid[1] = (d_y_max + d_y_min) / 2.0;
    d_centroid[2] = (d_z_max + d_z_min) / 2.0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Tuple constructor.
 *
 * \param bounds Tuple containing {x_min, y_min, z_min, x_max, y_max, z_max}.
 */
Box::Box( const EntityId global_id,
	  const int owner_rank, 
	  const Teuchos::Tuple<double,6>& bounds )
    : d_global_id( global_id )
    , d_owner_rank( owner_rank )
    , d_x_min( bounds[0] )
    , d_y_min( bounds[1] )
    , d_z_min( bounds[2] )
    , d_x_max( bounds[3] )
    , d_y_max( bounds[4] )
    , d_z_max( bounds[5] )
    , d_byte_size( sizeof(EntityId) + sizeof(int) + 6*sizeof(double) )
{
    DTK_REQUIRE( d_x_min <= d_x_max );
    DTK_REQUIRE( d_y_min <= d_y_max );
    DTK_REQUIRE( d_z_min <= d_z_max );
    d_centroid[0] = (d_x_max + d_x_min) / 2.0;
    d_centroid[1] = (d_y_max + d_y_min) / 2.0;
    d_centroid[2] = (d_z_max + d_z_min) / 2.0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor.
 */
Box::~Box()
{ /* ... */ }

//---------------------------------------------------------------------------//
// Return a string indicating the derived entity type.
std::string Box::entityType() const
{
    return std::string("DTK Box");
}

//---------------------------------------------------------------------------//
// Get the unique global identifier for the entity.
EntityId Box::id() const
{
    return d_global_id;
}
    
//---------------------------------------------------------------------------//
// Get the parallel rank that owns the entity.
int Box::ownerRank() const
{
    return d_owner_rank;
}

//---------------------------------------------------------------------------//
// Return the physical dimension of the entity.
int Box::physicalDimension() const
{
    return 3;
}

//---------------------------------------------------------------------------//
// Return the parametric dimension of the entity.
int Box::parametricDimension() const
{
    return 3;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the volume of the box.
 *
 * \return Return the volume of the box.
 */
double Box::measure() const
{
    return (d_x_max-d_x_min)*(d_y_max-d_y_min)*(d_z_max-d_z_min);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the centroid of the box.
 *
 * \return The centroid coordinates.
 */
void Box::centroid( Teuchos::ArrayView<const double>& centroid ) const
{
    centroid = Teuchos::ArrayView<const double>(d_centroid,3);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the bounding box around the box.
 *
 * \return The bounding box around the box.
 */
void Box::boundingBox( Box& bounding_box ) const
{
    bounding_box = *this;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Map a point to the reference space of an entity. Return the
 * parameterized point.
 */
void Box::safeguardMapToReferenceFrame(
	const Teuchos::ParameterList& parameters,
	const Teuchos::ArrayView<const double>& point,
	MappingStatus& status ) const
{
    status.mappingSucceeded();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Map a point to the reference space of an entity. Return the
 */
void Box::mapToReferenceFrame( 
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& point,
    const Teuchos::ArrayView<double>& reference_point,
    MappingStatus& status ) const
{
    reference_point.assign( point );
    status.mappingSucceeded();
}

//---------------------------------------------------------------------------//
/*!  
 * \brief Determine if a reference point is in the parameterized space of
 * an entity.
 */
bool Box::checkPointInclusion( 
    const Teuchos::ParameterList& parameters,
    const Teuchos::ArrayView<const double>& reference_point ) const
{
    DTK_REQUIRE( 3 == reference_point.size() );

    double tolerance = parameters.get<double>("Inclusion Tolerance");

    double x_tol = (d_x_max - d_x_min)*tolerance;
    double y_tol = (d_y_max - d_y_min)*tolerance;
    double z_tol = (d_z_max - d_z_min)*tolerance;

    if ( reference_point[0] >= d_x_min - x_tol &&
	 reference_point[1] >= d_y_min - y_tol &&
	 reference_point[2] >= d_z_min - z_tol &&
	 reference_point[0] <= d_x_max + x_tol &&
	 reference_point[1] <= d_y_max + y_tol &&
	 reference_point[2] <= d_z_max + z_tol )
    {
	return true;
    }

    return false;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Map a reference point to the physical space of an entity.
 */
void Box::mapToPhysicalFrame( 
    const Teuchos::ArrayView<const double>& reference_point,
    const Teuchos::ArrayView<double>& point ) const
{
    point.assign( reference_point );
}

//---------------------------------------------------------------------------//
// Get the size of the serialized entity in bytes.
std::size_t Box::byteSize() const
{
    return d_byte_size;
}

//---------------------------------------------------------------------------//
// Serialize the entity into a buffer.
void Box::serialize( const Teuchos::ArrayView<char>& buffer ) const
{
    DTK_REQUIRE( Teuchos::as<std::size_t>(buffer.size()) == d_byte_size );
    DataSerializer serializer;
    serializer.setBuffer( buffer );
    serializer << d_global_id << d_owner_rank
	       << d_x_min << d_y_min << d_z_min
	       << d_x_max << d_y_max << d_z_max;
}

//---------------------------------------------------------------------------//
// Deserialize an entity from a buffer.
void Box::deserialize( const Teuchos::ArrayView<const char>& buffer )
{
    // Unpack the data.
    DTK_REQUIRE( Teuchos::as<std::size_t>(buffer.size()) == d_byte_size );
    DataDeserializer deserializer;
    Teuchos::ArrayView<char> buffer_nonconst(
	const_cast<char*>(buffer.getRawPtr()), buffer.size() );
    deserializer.setBuffer( buffer_nonconst );
    deserializer >> d_global_id >> d_owner_rank
		 >> d_x_min >> d_y_min >> d_z_min
		 >> d_x_max >> d_y_max >> d_z_max;

    // Build the centroid.
    d_centroid[0] = (d_x_max + d_x_min) / 2.0;
    d_centroid[1] = (d_y_max + d_y_min) / 2.0;
    d_centroid[2] = (d_z_max + d_z_min) / 2.0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Static function for box intersection test.
 *
 * \param box_A box A.
 *
 * \param box_B box B.
 *
 * \return Return true if the boxes intersect. False if they do not.
 */
bool Box::checkForIntersection( const Box& box_A,
				const Box& box_B )
{
    Teuchos::Tuple<double,6> bounds_A = box_A.getBounds();
    Teuchos::Tuple<double,6> bounds_B = box_B.getBounds();

    return !( ( bounds_A[0] > bounds_B[3] || bounds_A[3] < bounds_B[0] ) ||
	      ( bounds_A[1] > bounds_B[4] || bounds_A[4] < bounds_B[1] ) ||
	      ( bounds_A[2] > bounds_B[5] || bounds_A[5] < bounds_B[2] ) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Static function for box intersection. Return false if they do not
 * intersect. 
 *
 * \param box_A box A.
 *
 * \param box_B box B.
 *
 * \param box intersection A box that is equivalent to the intersection of box
 * A and box B. Box A and B can be provided in any order (the intersection of
 * box A with box B is equal to the intersection of box B with box A).
 *
 * \return Return true if the boxes intersect. False if they do not.
 */
bool Box::intersectBoxes( const Box& box_A,
			  const Box& box_B,
			  Box& box_intersection)
{
    // Check for intersection.
    if ( !checkForIntersection(box_A,box_B) )
    {
	return false;
    }

    Teuchos::Tuple<double,6> bounds_A = box_A.getBounds();
    Teuchos::Tuple<double,6> bounds_B = box_B.getBounds();

    double x_min, y_min, z_min, x_max, y_max, z_max;

    // Get overlap in X.
    if ( bounds_A[0] > bounds_B[0] )
    {
	x_min = bounds_A[0];
    }
    else
    {
	x_min = bounds_B[0];
    }
    if ( bounds_A[3] > bounds_B[3] )
    {
	x_max = bounds_B[3];
    }
    else
    {
	x_max = bounds_A[3];
    }

    // Get overlap in Y.
    if ( bounds_A[1] > bounds_B[1] )
    {
	y_min = bounds_A[1];
    }
    else
    {
	y_min = bounds_B[1];
    }
    if ( bounds_A[4] > bounds_B[4] )
    {
	y_max = bounds_B[4];
    }
    else
    {
	y_max = bounds_A[4];
    }

    // Get overlap in Z.
    if ( bounds_A[2] > bounds_B[2] )
    {
	z_min = bounds_A[2];
    }
    else
    {
	z_min = bounds_B[2];
    }
    if ( bounds_A[5] > bounds_B[5] )
    {
	z_max = bounds_B[5];
    }
    else
    {
	z_max = bounds_A[5];
    }

    box_intersection = Box( dtk_invalid_entity_id, box_A.ownerRank(),
			    x_min, y_min, z_min, 
			    x_max, y_max, z_max );
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Static function for box union. 
 *
 * \param box_A box A.
 *
 * \param box_B box B.
 *
 * \param box_unition A box that is equivalent to the union of box A
 * and box B. Box A and B can be provided in any order (the union of
 * box A with box B is equal to the union of box B with box A).
 */
void Box::uniteBoxes( const Box& box_A,
		      const Box& box_B,
		      Box& box_union)
{
    Teuchos::Tuple<double,6> bounds_A = box_A.getBounds();
    Teuchos::Tuple<double,6> bounds_B = box_B.getBounds();

    double x_min, y_min, z_min, x_max, y_max, z_max;

    // Get overlap in X.
    if ( bounds_A[0] < bounds_B[0] )
    {
	x_min = bounds_A[0];
    }
    else
    {
	x_min = bounds_B[0];
    }
    if ( bounds_A[3] < bounds_B[3] )
    {
	x_max = bounds_B[3];
    }
    else
    {
	x_max = bounds_A[3];
    }

    // Get overlap in Y.
    if ( bounds_A[1] < bounds_B[1] )
    {
	y_min = bounds_A[1];
    }
    else
    {
	y_min = bounds_B[1];
    }
    if ( bounds_A[4] < bounds_B[4] )
    {
	y_max = bounds_B[4];
    }
    else
    {
	y_max = bounds_A[4];
    }

    // Get overlap in Z.
    if ( bounds_A[2] < bounds_B[2] )
    {
	z_min = bounds_A[2];
    }
    else
    {
	z_min = bounds_B[2];
    }
    if ( bounds_A[5] < bounds_B[5] )
    {
	z_max = bounds_B[5];
    }
    else
    {
	z_max = bounds_A[5];
    }

    box_union = Box( dtk_invalid_entity_id, box_A.ownerRank(),
		     x_min, y_min, z_min, 
		     x_max, y_max, z_max );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Addition operator. Adding two boxes together is the same as
 * computing their union.
 */
Box operator+( const Box& box_1, const Box& box_2 )
{
    Box union_box;
    Box::uniteBoxes( box_1, box_2, union_box );
    return union_box;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Print the box description to an ostream.
 *
 * \return The ostream.
 */
std::ostream& operator<< (std::ostream& os,const DataTransferKit::Box& b)
{
  Teuchos::Tuple<double,6> bounds = b.getBounds();

  os << "Box: d_global_id=" << b.id()
     << ",d_owner_rank=" << b.ownerRank()
     << ",d_x_min=" << bounds[0] 
     << ",d_y_min=" << bounds[1]
     << ",d_z_min=" << bounds[2]
     << ",d_x_max=" << bounds[3]
     << ",d_y_max=" << bounds[4]
     << ",d_z_max=" << bounds[5];

  return os;
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_Box.cpp
//---------------------------------------------------------------------------//

