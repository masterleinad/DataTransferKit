/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_DETAILS_SPLINE_OPERATOR_IMPL_HPP
#define DTK_DETAILS_SPLINE_OPERATOR_IMPL_HPP

#include <ArborX.hpp>
#include <ArborX_DetailsKokkosExt.hpp> // ArithmeticTraits
#include <DTK_CompactlySupportedRadialBasisFunctions.hpp>
#include <DTK_DetailsSVDImpl.hpp>

namespace DataTransferKit
{
namespace Details
{
template <typename DeviceType>
struct SplineOperatorImpl
{
    using ExecutionSpace = typename DeviceType::execution_space;

    static Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType>
    makeKNNQueries( typename Kokkos::View<Coordinate **, DeviceType>::const_type
                        target_points,
                    unsigned int n_neighbors )
    {
        auto const n_points = target_points.extent( 0 );
        Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
            "queries", n_points );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "setup_queries" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            KOKKOS_LAMBDA( int i ) {
                queries( i ) = nearest(
                    ArborX::Point{{target_points( i, 0 ), target_points( i, 1 ),
                                   target_points( i, 2 )}},
                    n_neighbors );
            } );
        Kokkos::fence();
        return queries;
    }

    static Kokkos::View<double *, DeviceType> computeTargetValues(
        Kokkos::View<int const *, DeviceType> offset,
        Kokkos::View<double const *, DeviceType> polynomial_coeffs,
        Kokkos::View<double const *, DeviceType> source_values )
    {
        auto const n_target_points = offset.extent_int( 0 ) - 1;
        Kokkos::View<double *, DeviceType> target_values(
            std::string( "target_" ) + source_values.label(), n_target_points );

        Kokkos::parallel_for(
            DTK_MARK_REGION( "compute_values" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_target_points ),
            KOKKOS_LAMBDA( const int i ) {
                target_values( i ) = 0.;
                for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                    target_values( i ) +=
                        polynomial_coeffs( j ) * source_values( j );
            } );
        Kokkos::fence();

        return target_values;
    }

    static Kokkos::View<double *, DeviceType>
    computeRadius( Kokkos::View<Coordinate const **, DeviceType> needed_source_points,
		   Kokkos::View<Coordinate const **, DeviceType> target_points, 
                   Kokkos::View<int const *, DeviceType> offset )
    {
        unsigned int const n_target_points = offset.extent( 0 ) - 1;
        Kokkos::View<double *, DeviceType> radius( "radius",
                                                   needed_source_points.extent( 0 ) );

        Kokkos::parallel_for(
            DTK_MARK_REGION( "compute_radius" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_target_points ),
            KOKKOS_LAMBDA( int const i ) {
                // If the source point and the target point are at the same
                // position, the radius will be zero. This is a problem since we
                // divide by the radius in the calculation of the radial basis
                // function. To avoid this problem, the radius has a minimal
                // positive value.
                double distance =
                    10. * KokkosExt::ArithmeticTraits::epsilon<double>::value;
                for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                {
                    double new_distance = ArborX::Details::distance(
                        ArborX::Point{{needed_source_points( j, 0 ),
                                       needed_source_points( j, 1 ),
                                       needed_source_points( j, 2 )}},
                        ArborX::Point{{target_points( i, 0 ),
			               target_points( i, 1 ),
				       target_points( i, 2 )}});

                    if ( new_distance > distance )
                        distance = new_distance;
                }
                // If a point is exactly on the boundary of the compact domain,
                // its weight will be zero so we need to make sure that no point
                // is exactly on the boundary.
                for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                    radius( j ) = 1.1 * distance;
            } );

        return radius;
    }

    // We need the third argument because otherwise the compiler cannot do the
    // template deduction. For some unknown reason, explicitly choosing the
    // value of the template parameter does not work.
    template <typename RBF>
    static Kokkos::View<double *, DeviceType>
    computeWeights( Kokkos::View<Coordinate const **, DeviceType> needed_source_points,
		    Kokkos::View<Coordinate const **, DeviceType> target_points,
                    Kokkos::View<double const *, DeviceType> radius,
		    Kokkos::View<int const *, DeviceType> offset,
                    RBF const & )
    {
        auto const n_target_points = target_points.extent( 0 );

        DTK_REQUIRE( target_points.extent_int( 1 ) == 3 );

        // The argument of rbf is a distance because we have changed the
        // coordinate system such the target point is the origin of the new
        // coordinate system.
        Kokkos::View<double *, DeviceType> phi( "weights", needed_source_points.extent(0));
        Kokkos::parallel_for(
            DTK_MARK_REGION( "compute_weights" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_target_points ),
            KOKKOS_LAMBDA( int i ) {
	     for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                {
                RadialBasisFunction<RBF> rbf( radius( j ) );
		std::cout << "acessing " << i << " " << j << " out of " << needed_source_points.extent(0) << std::endl;
                phi( j ) = rbf( ArborX::Details::distance(
                    ArborX::Point{{needed_source_points( i, 0 ), needed_source_points( i, 1 ),
                                   needed_source_points( i, 2 )}},
                    ArborX::Point{{target_points( j, 0), target_points(j,1), 
		                   target_points( j, 2)}} ) );
		}} );
        Kokkos::fence();
        return phi;
    }

    template <typename PolynomialBasis>
    static Kokkos::View<double *, DeviceType>
    computeVandermonde( Kokkos::View<Coordinate const **, DeviceType> points,
                        PolynomialBasis const &polynomial_basis )
    {
        auto const n_points = points.extent( 0 );
        auto constexpr size_polynomial_basis = PolynomialBasis::size;
        Kokkos::View<double *, DeviceType> p(
            "vandermonde", n_points * size_polynomial_basis );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "compute_polynomial_basis" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            KOKKOS_LAMBDA( int i ) {
                auto const tmp = polynomial_basis( ArborX::Point{
                    {points( i, 0 ), points( i, 1 ), points( i, 2 )}} );
                for ( int j = 0; j < size_polynomial_basis; ++j )
                    p( i * size_polynomial_basis + j ) = tmp[j];
            } );
        return p;
    }
};

} // end namespace Details
} // end namespace DataTransferKit

#endif
