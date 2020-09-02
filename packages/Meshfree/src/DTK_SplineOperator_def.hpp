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
#include <DTK_DBC.hpp>
#include <DTK_DetailsSplineOperatorImpl.hpp>
#include <DTK_DetailsNearestNeighborOperatorImpl.hpp> // fetch
#include <DTK_PolynomialMatrix.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosBlockGmresSolMgr.hpp>

namespace DataTransferKit
{

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
SplineOperator<DeviceType, CompactlySupportedRadialBasisFunction,
                           PolynomialBasis>::
    SplineOperator(
        MPI_Comm comm,
        Kokkos::View<Coordinate const **, DeviceType> source_points,
        Kokkos::View<Coordinate const **, DeviceType> target_points )
    : _comm( comm )
    , _n_source_points( source_points.extent( 0 ) )
    , _offset( "offset", 0 )
    , _ranks( "ranks", 0 )
    , _indices( "indices", 0 )
    , _coeffs( "polynomial_coefficients", 0 )
{
    using global_ordinal_type = typename Tpetra::Vector<>::global_ordinal_type;
    using local_ordinal_type = typename Tpetra::Vector<>::local_ordinal_type;
    using scalar_type = typename Tpetra::Vector<>::scalar_type;
    constexpr global_ordinal_type indexBase = 0;

    DTK_REQUIRE( source_points.extent_int( 1 ) ==
                 target_points.extent_int( 1 ) );
    // FIXME for now let's assume 3D
    DTK_REQUIRE( source_points.extent_int( 1 ) == 3 );

    auto teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(comm));

      int n_global_target_points;
    int n_local_target_points = target_points.extent(0);
    MPI_Allreduce(&n_local_target_points, &n_global_target_points, 1, MPI_INT, MPI_SUM, comm);
    _destination_map = Teuchos::rcp (new Tpetra::Map<> (n_global_target_points, n_local_target_points, indexBase, teuchos_comm));
    _destination = Teuchos::rcp( new VectorType(_destination_map, 1));

        int n_global_source_points;
    int n_local_source_points = source_points.extent(0);
    MPI_Allreduce(&n_local_source_points, &n_global_source_points, 1, MPI_INT, MPI_SUM, comm);
    _source_map = Teuchos::rcp (new Tpetra::Map<> (n_global_source_points, n_local_source_points, indexBase, teuchos_comm));
    _source = Teuchos::rcp( new VectorType(_source_map, 1));


    //------------Build the matrices-----

    // Build distributed search tree over the source points.
    ArborX::DistributedSearchTree<DeviceType> search_tree( _comm,
                                                           source_points );
    DTK_CHECK( !search_tree.empty() );

    // For each source point, query the n_neighbors points closest to the
    // source.
    auto source_queries =
        Details::SplineOperatorImpl<DeviceType>::makeKNNQueries(
            source_points, PolynomialBasis::size );

    // Perform the actual search.
    search_tree.query( source_queries, _indices, _offset, _ranks );

    // Retrieve the coordinates of all source points that met the predicates.
    // NOTE: This is the last collective.
    auto needed_source_points_M = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, _ranks, _indices, source_points );

     // Create the P matrix.
    constexpr int DIM=3;
    int offset = DIM + 1;
    auto P_vec =
        Teuchos::rcp(new VectorType(_source_map, offset ));
    for ( unsigned i = 0; i < _n_source_points; ++i )
    {
        const auto global_id = _source_map->getGlobalElement(i);
        P_vec->replaceGlobalValue(global_id, 0, 1.0 );
        for ( int d = 0; d < DIM; ++d )
            P_vec->replaceGlobalValue(global_id, d+1, source_points(i,d) );
    }
    auto d_P =Teuchos::rcp( new PolynomialMatrix(P_vec,_source_map,_source_map) );

    // Create the M matrix

    std::cout << "before radius" << std::endl;

    // To build the radial basis function, we need to define the radius of the
    // radial basis function. Since we use kNN, we need to compute the radius.
    // We only need the coordinates of the source points because of the
    // transformation of the coordinates.
    auto radius =
        Details::SplineOperatorImpl<DeviceType>::computeRadius(
            needed_source_points_M, source_points, _offset );

    std::cout << "before weights" << std::endl;

    // Build phi (weight matrix)
    auto phi_M =
        Details::SplineOperatorImpl<DeviceType>::computeWeights(
            source_points, source_points, radius, _offset, CompactlySupportedRadialBasisFunction() );

    // build matrix
    _crs_matrix = Teuchos::rcp(new Tpetra::CrsMatrix<>(_source_map, _n_source_points));

    std::cout << "before matrix" << std::endl;

     int local_cumulative_points = 0;
     MPI_Scan(&_n_source_points, &local_cumulative_points, 1, MPI_INT, MPI_SUM, comm);

     int n_processes = 0;
     MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

     cumulative_points_per_process.resize(n_processes+1);
     MPI_Allgather(&local_cumulative_points, 1, MPI_INT, &(cumulative_points_per_process[1]), 1, MPI_INT, comm);

     for (local_ordinal_type i=0; i< n_local_target_points; ++i)
             for ( int j = _offset( i ); j < _offset( i + 1 ); ++j )
                {
			        const auto global_id = _source_map->getGlobalElement(i);
                _crs_matrix->insertGlobalValues(global_id, 
				                Teuchos::tuple<global_ordinal_type>(cumulative_points_per_process[_ranks(j)]+_indices(j)),
						Teuchos::tuple<scalar_type>(phi_M(j)));
                }

    // Tell the sparse matrix that we are done adding entries to it.
    _crs_matrix->fillComplete ();

    // N matrix

     // For each source point, query the n_neighbors points closest to the
    // source.
    auto target_queries =
        Details::SplineOperatorImpl<DeviceType>::makeKNNQueries(
            target_points, PolynomialBasis::size );

    // Perform the actual search.
    search_tree.query( target_queries, target_indices, target_offset, target_ranks );

    Kokkos::View<int *, DeviceType> target_ranks;

    // Retrieve the coordinates of all source points that met the predicates.
    // NOTE: This is the last collective.
    auto needed_source_points_Q = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, target_ranks, target_indices, source_points );

    auto target_radius =
        Details::SplineOperatorImpl<DeviceType>::computeRadius(
            needed_source_points_Q, target_points, target_offset );

     auto phi_N =
        Details::SplineOperatorImpl<DeviceType>::computeWeights(
            source_points, needed_source_points_Q, target_radius, target_offset, CompactlySupportedRadialBasisFunction() );


   //....
   
      // Create the Q matrix.
     {
    constexpr int DIM=3;
    int offset = DIM + 1;
    auto Q_vec =
        Teuchos::rcp(new VectorType(_destination_map, offset ));
    for ( int i = 0; i < n_local_target_points; ++i )
    {
	const auto global_id = _destination_map->getGlobalElement(i);
        Q_vec->replaceGlobalValue( global_id, 0, 1.0 );
        for ( int d = 0; d < DIM; ++d )
        {
            Q_vec->replaceGlobalValue(
                global_id, d+1, target_points(i,d) );
        }
    }
    auto d_Q =Teuchos::rcp( new PolynomialMatrix(Q_vec,_destination_map,_destination_map) );
     }
}

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
void SplineOperator<
    DeviceType, CompactlySupportedRadialBasisFunction, PolynomialBasis>::
    apply( Kokkos::View<double const *, DeviceType> source_values,
           Kokkos::View<double *, DeviceType> target_values ) const
{
    std::cout << "start apply" << std::endl;

    // Precondition: check that the source and the target are properly sized
    DTK_REQUIRE( source_values.extent( 0 ) == _n_source_points );
    DTK_REQUIRE( target_values.extent( 0 ) == _offset.extent( 0 ) - 1 );

    // Retrieve values for all source points
    source_values = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, _ranks, _indices, source_values );

    // copy source_values to source
    for (unsigned int i=0; i<_ranks.extent(0); ++i)
    {
       const auto global_id = _source_map->getGlobalElement(i);
      _source->replaceGlobalValue(global_id,0,source_values(i));
    }

    auto problem = Teuchos::rcp (new Belos::LinearProblem<ScalarType,VectorType,OperatorType>(_crs_matrix, _source, _destination));
    problem->setProblem();

    Teuchos::RCP<Teuchos::ParameterList> params;
    // params->set(...);
    Belos::BlockGmresSolMgr<ScalarType,VectorType,OperatorType> solver( problem, params );
    auto ret = solver.solve();
    (void) ret;
    DTK_REQUIRE(ret == Belos::Converged);
    auto solution = problem->getLHS();

    // copy solution to target_values
    for(unsigned int i=0; i<target_values.size(); ++i)
	    target_values(i) = solution->getLocalViewHost()(i,0);    
}

} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_SPLINE_OPERATOR_INSTANT( NODE )                      \
    template class SplineOperator<typename NODE::device_type>;     \
    template class SplineOperator<                                 \
        typename NODE::device_type, Wendland<0>,                               \
        MultivariatePolynomialBasis<Quadratic, 3>>;

#endif
