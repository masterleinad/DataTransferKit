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

#ifndef DTK_SPLINE_OPERATOR_DEF_HPP
#define DTK_SPLINE_OPERATOR_DEF_HPP

#include <ArborX.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <DTK_DBC.hpp>
#include <DTK_DetailsNearestNeighborOperatorImpl.hpp> // fetch
#include <DTK_DetailsSplineOperatorImpl.hpp>
#include <DTK_PolynomialMatrix.hpp>
#include <DTK_SplineProlongationOperator.hpp>

#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Thyra_DefaultAddedLinearOp.hpp>
#include <Thyra_DefaultMultipliedLinearOp.hpp>
#include <Thyra_DefaultScaledAdjointLinearOp.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

namespace DataTransferKit
{

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
Teuchos::RCP<
    typename SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
                            PolynomialBasis>::Operator>
SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
               PolynomialBasis>::
    buildBasisOperator(
        Teuchos::RCP<const Map> domain_map, Teuchos::RCP<const Map> range_map,
        Kokkos::View<Coordinate const **, DeviceType> source_points,
        Kokkos::View<Coordinate const **, DeviceType> target_points,
        int const knn )
{
    auto teuchos_comm = domain_map->getComm();
    auto teuchos_mpi_comm =
        Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>( teuchos_comm );
    MPI_Comm comm = ( *teuchos_mpi_comm->getRawMpiComm() )();

    int const num_source_points = source_points.extent( 0 );
    int const num_points = target_points.extent( 0 );

    Kokkos::View<int *, DeviceType> offset( "offset", 0 );
    Kokkos::View<int *, DeviceType> ranks( "ranks", 0 );
    Kokkos::View<int *, DeviceType> indices( "indices", 0 );
    ArborX::DistributedSearchTree<DeviceType> distributed_tree( comm,
                                                                source_points );
    DTK_CHECK( !distributed_tree.empty() );

    // Perform the actual search.
    auto queries = Details::SplineOperatorImpl<DeviceType>::makeKNNQueries(
        target_points, knn );
    distributed_tree.query( queries, indices, offset, ranks );

    // Retrieve the coordinates of all points that met the predicates.
    auto source_points_with_halo =
        Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
            comm, ranks, indices, source_points );

    if ( source_points == target_points )
    {
        _ranks = ranks;
        _indices = indices;
    }

    // To build the radial basis function, we need to define the radius of
    // the radial basis function. Since we use kNN, we need to compute the
    // radius. We only need the coordinates of the source points because of
    // the transformation of the coordinates.
    auto radius = Details::SplineOperatorImpl<DeviceType>::computeRadius(
        source_points_with_halo, target_points, offset );

    // Build phi (weight matrix)
    auto phi = Details::SplineOperatorImpl<DeviceType>::computeWeights(
        source_points_with_halo, target_points, radius, offset,
        CompactlySupportedRadialBasisFunction() );

    // Build matrix
    auto row_map = range_map;
    auto crs_matrix = Teuchos::rcp( new CrsMatrix( row_map, knn ) );

    int rank_offset = 0;
    MPI_Scan( &num_source_points, &rank_offset, 1, MPI_INT, MPI_SUM, comm );

    int comm_size = teuchos_comm->getSize();

    std::vector<int> cumulative_points_per_process( comm_size + 1 );
    MPI_Allgather( &rank_offset, 1, MPI_INT,
                   &( cumulative_points_per_process[1] ), 1, MPI_INT, comm );

    for ( LO i = 0; i < num_points; ++i )
        for ( int j = offset( i ); j < offset( i + 1 ); ++j )
        {
            const auto global_id = row_map->getGlobalElement( i );
            crs_matrix->insertGlobalValues(
                global_id,
                Teuchos::tuple<GO>( cumulative_points_per_process[ranks( j )] +
                                    indices( j ) ),
                Teuchos::tuple<SC>( phi( j ) ) );
        }

    crs_matrix->fillComplete( domain_map, range_map );
    DTK_ENSURE( crs_matrix->isFillComplete() );

    return crs_matrix;
}

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
Teuchos::RCP<
    typename SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
                            PolynomialBasis>::Operator>
SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
               PolynomialBasis>::
    buildPolynomialOperator(
        Teuchos::RCP<const Map> domain_map, Teuchos::RCP<const Map> range_map,
        Kokkos::View<Coordinate const **, DeviceType> points )
{
    const int n = points.extent( 0 );
    const int spatial_dim = points.extent( 1 );

    DTK_REQUIRE( spatial_dim == 3 );

    auto v = Teuchos::rcp( new Vector( range_map, spatial_dim + 1 ) );

    for ( LO i = 0; i < n; ++i )
    {
        const auto global_id = range_map->getGlobalElement( i );
        v->replaceGlobalValue( global_id, 0, 1.0 );
        for ( int d = 0; d < spatial_dim; ++d )
            v->replaceGlobalValue( global_id, d + 1, points( i, d ) );
    }
    return Teuchos::rcp(
        new PolynomialMatrix<SC, LO, GO, NO>( v, domain_map, range_map ) );
}

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
               PolynomialBasis>::
    SplineOperator(
        MPI_Comm comm,
        Kokkos::View<Coordinate const **, DeviceType> source_points,
        Kokkos::View<Coordinate const **, DeviceType> target_points )
    : _comm( comm )
{
    DTK_REQUIRE( source_points.extent_int( 1 ) ==
                 target_points.extent_int( 1 ) );
    // FIXME for now let's assume 3D
    DTK_REQUIRE( source_points.extent_int( 1 ) == 3 );
    constexpr int spatial_dim = 3;

    constexpr int knn = PolynomialBasis::size;

    // Step 0: build source and target maps
    auto teuchos_comm = Teuchos::rcp( new Teuchos::MpiComm<int>( comm ) );
    auto source_map = Teuchos::rcp(
        new Map( Teuchos::OrdinalTraits<GO>::invalid(),
                 source_points.extent( 0 ), 0 /*indexBase*/, teuchos_comm ) );
    auto target_map = Teuchos::rcp(
        new Map( Teuchos::OrdinalTraits<GO>::invalid(),
                 target_points.extent( 0 ), 0 /*indexBase*/, teuchos_comm ) );

    // Step 1: build matrices
    GO prolongation_offset = teuchos_comm->getRank() ? 0 : spatial_dim + 1;
    S = Teuchos::rcp( new SplineProlongationOperator<SC, LO, GO, NO>(
        prolongation_offset, source_map ) );
    auto prolongation_map = S->getRangeMap();

    // Build distributed search tree over the source points.
    // NOTE: M is not the M from the paper, but an extended size block matrix
    M = buildBasisOperator( prolongation_map, prolongation_map, source_points,
                            source_points, knn );
    P = buildPolynomialOperator( prolongation_map, prolongation_map,
                                 source_points );
    N = buildBasisOperator( prolongation_map, target_map, source_points,
                            target_points, knn );
    Q = buildPolynomialOperator( prolongation_map, target_map, target_points );

    // Step 3: build Thyra operator: A = (Q + N)*[(P + M + P^T)^-1]*S
    auto thyraWrapper = []( Teuchos::RCP<const Operator> &op ) {
        auto thyra_range_vector_space =
            Thyra::createVectorSpace<SC>( op->getRangeMap() );
        auto thyra_domain_vector_space =
            Thyra::createVectorSpace<SC>( op->getDomainMap() );
        using ThyraOperator = Thyra::TpetraLinearOp<SC, LO, GO, NO>;
        auto thyra_op = Teuchos::rcp( new ThyraOperator() );
        Teuchos::rcp_const_cast<ThyraOperator>( thyra_op )
            ->constInitialize( thyra_range_vector_space,
                               thyra_domain_vector_space, op );
        return thyra_op;
    };

    auto thyra_S = thyraWrapper( S );
    auto thyra_M = thyraWrapper( M );
    auto thyra_N = thyraWrapper( N );
    auto thyra_P = thyraWrapper( P );
    auto thyra_Q = thyraWrapper( Q );

    // Create a transpose of P.
    Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyra_P_T =
        Thyra::transpose<SC>( thyra_P );

    // Create a composite operator C = (P + M + P^T)
    Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyra_PpM =
        Thyra::add<SC>( thyra_P, thyra_M );
    Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyra_C =
        Thyra::add<SC>( thyra_PpM, thyra_P_T );

    // If we didnt get stratimikos parameters from the input list, create
    // some here.
    Teuchos::RCP<Teuchos::ParameterList> d_stratimikos_list;
    if ( Teuchos::is_null( d_stratimikos_list ) )
    {
        d_stratimikos_list = Teuchos::parameterList( "Stratimikos" );
        // clang-format off
        Teuchos::updateParametersFromXmlString(
            "<ParameterList name=\"Stratimikos\">"
              "<Parameter name=\"Linear Solver Type\"  type=\"string\" value=\"Belos\"/>"
              "<Parameter name=\"Preconditioner Type\" type=\"string\" value=\"None\"/>"
            "</ParameterList>",
            d_stratimikos_list.ptr() );
        // clang-format on
    }

    // Create the inverse of the composite operator C.
    Stratimikos::DefaultLinearSolverBuilder builder;
    builder.setParameterList( d_stratimikos_list );
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<SC>> factory =
        Thyra::createLinearSolveStrategy( builder );
    Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyra_C_inv =
        Thyra::inverse<SC>( *factory, thyra_C );

    // Create the composite operator B = (Q + N);
    Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyra_B =
        Thyra::add<SC>( thyra_Q, thyra_N );

    // Create the coupling matrix A = (B * C^-1 * S).
    _thyra_operator = Thyra::multiply<SC>( thyra_B, thyra_C_inv, thyra_S );
    DTK_ENSURE( Teuchos::nonnull( _thyra_operator ) );
}

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
void SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
                    PolynomialBasis>::
    apply( Kokkos::View<double const *, DeviceType> source_values,
           Kokkos::View<double *, DeviceType> target_values ) const
{
    std::cout << "start apply" << std::endl;

    // Precondition: check that the source and the target are properly sized
    // DTK_REQUIRE( source_values.extent( 0 ) == _n_source_points );
    DTK_REQUIRE( target_values.extent( 0 ) == target_offset.extent( 0 ) - 1 );

    auto source = Teuchos::rcp( new Vector( S->getDomainMap(), 1 ) );
    auto destination = Teuchos::rcp( new Vector( N->getRangeMap(), 1 ) );

    // Retrieve values for all source points
    source_values = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, _ranks, _indices, source_values );

    auto domain_map = M->getDomainMap();

    // copy source_values to source
    for ( unsigned int i = 0; i < _ranks.extent( 0 ); ++i )
    {
        const auto global_id = domain_map->getGlobalElement( i );
        source->replaceGlobalValue( global_id, 0, source_values( i ) );
    }

    auto thyra_X = Thyra::createMultiVector<SC>( source );
    auto thyra_Y = Thyra::createMultiVector<SC>( destination );
    _thyra_operator->apply( Thyra::NOTRANS, *thyra_X, thyra_Y.ptr(), 1, 0 );

    // copy solution to target_values
    for ( unsigned int i = 0; i < target_values.size(); ++i )
        target_values( i ) = destination->getLocalViewHost()( i, 0 );
}

} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_SPLINE_OPERATOR_INSTANT( NODE )                                    \
    template class SplineOperator<typename NODE::device_type>;                 \
    template class SplineOperator<typename NODE::device_type, Wendland<0>,     \
                                  MultivariatePolynomialBasis<Quadratic, 3>>;

#endif
