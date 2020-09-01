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

    DTK_REQUIRE( source_points.extent_int( 1 ) ==
                 target_points.extent_int( 1 ) );
    // FIXME for now let's assume 3D
    DTK_REQUIRE( source_points.extent_int( 1 ) == 3 );

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
    auto needed_source_points = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, _ranks, _indices, source_points );

    int local_cumulative_points = 0;
    MPI_Scan(&_n_source_points, &local_cumulative_points, 1, MPI_INT, MPI_SUM, comm);

    int n_processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    cumulative_points_per_process.resize(n_processes+1);
    MPI_Allgather(&local_cumulative_points, 1, MPI_INT, &(cumulative_points_per_process[1]), 1, MPI_INT, comm);

    // collect all the points we need locally and their global index
    std::vector<global_ordinal_type> required_indices;
    required_indices.reserve(_ranks.size());
    for (unsigned int i=0; i<_ranks.extent(0); ++i)
      required_indices.push_back(cumulative_points_per_process[_ranks(i)]+_indices(i));

    std::sort(required_indices.begin(), required_indices.end());
    required_indices.erase(std::unique( required_indices.begin(), required_indices.end()), required_indices.end());

    Kokkos::View<global_ordinal_type*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> kokkos_required_indices(required_indices.data(), required_indices.size());

    Kokkos::View<global_ordinal_type *, DeviceType> kokkos_indices("kokkos_indices", required_indices.size());
    Kokkos::deep_copy(kokkos_indices, kokkos_required_indices);     
    
    // create a map
    auto teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(comm));
    _source_map = Teuchos::rcp( new Tpetra::Map<>(cumulative_points_per_process.back(), kokkos_indices, 0, teuchos_comm));  

    _source = Teuchos::rcp( new VectorType(_source_map, 1));

     // Create the P matrix.
    constexpr int DIM=3;
    int offset = DIM + 1;
    auto P_vec =
        Teuchos::rcp(new VectorType(_source_map, offset ));
    int di = 0;
    for ( unsigned i = 0; i < _n_source_points; ++i )
    {
        P_vec->replaceGlobalValue( cumulative_points_per_process[teuchos_comm->getRank()]+i, 0, 1.0 );
        di = DIM*i;
        for ( int d = 0; d < DIM; ++d )
        {
            P_vec->replaceGlobalValue(
                cumulative_points_per_process[teuchos_comm->getRank()]+i, d+1, source_points(i,d) );
        }
    }
    auto d_P =Teuchos::rcp( new PolynomialMatrix(P_vec,_source_map,_source_map) );



    std::cout << "before radius" << std::endl;

    // To build the radial basis function, we need to define the radius of the
    // radial basis function. Since we use kNN, we need to compute the radius.
    // We only need the coordinates of the source points because of the
    // transformation of the coordinates.
    auto radius =
        Details::SplineOperatorImpl<DeviceType>::computeRadius(
            source_points, target_points, _offset );

    std::cout << "before weights" << std::endl;

    // Build phi (weight matrix)
    auto phi =
        Details::SplineOperatorImpl<DeviceType>::computeWeights(
            source_points, target_points, radius, _offset, CompactlySupportedRadialBasisFunction() );

    // build matrix
    int n_global_target_points;
    int n_local_target_points = _offset.extent( 0 ) - 1;
    MPI_Allreduce(&n_local_target_points, &n_global_target_points, 1, MPI_INT, MPI_SUM, comm);
    constexpr global_ordinal_type indexBase = 0;
    _destination_map = Teuchos::rcp (new Tpetra::Map<> (n_global_target_points, n_local_target_points, indexBase, teuchos_comm));

    _destination = Teuchos::rcp( new VectorType(_destination_map, 1));

    _crs_matrix = Teuchos::rcp(new Tpetra::CrsMatrix<>(_source_map, _n_source_points));

    std::cout << "before matrix" << std::endl;

    // upper right matrix and lower left matrix
    
    for (local_ordinal_type i = 0; i < _n_source_points; ++i)
    {
	    _crs_matrix->insertGlobalValues(_n_source_points+i, 
			                    Teuchos::tuple<global_ordinal_type>(0,1,2,3), 
					    Teuchos::tuple<scalar_type> (1, source_points(i, 0), source_points(i,1), source_points(i,2)));
	    _crs_matrix->insertGlobalValues(0, Teuchos::tuple<global_ordinal_type>(i), Teuchos::tuple<scalar_type> (1));
            _crs_matrix->insertGlobalValues(1, Teuchos::tuple<global_ordinal_type>(i), Teuchos::tuple<scalar_type> (source_points(i,0)));
            _crs_matrix->insertGlobalValues(2, Teuchos::tuple<global_ordinal_type>(i), Teuchos::tuple<scalar_type> (source_points(i,1)));
            _crs_matrix->insertGlobalValues(3, Teuchos::tuple<global_ordinal_type>(i), Teuchos::tuple<scalar_type> (source_points(i,2)));
    }

    std::cout << "lower right" << std::endl;

    //lower right block
    for (local_ordinal_type i=0; i<_n_source_points; ++i)
    {
	    for (local_ordinal_type j=0; j<target_points.extent(0); ++j)
              _crs_matrix->insertGlobalValues(i, Teuchos::tuple<global_ordinal_type>(j),
		                                 Teuchos::tuple<scalar_type>(phi(i,j)));
    }

    std::cout << "fill complete" << std::endl;

    //const global_ordinal_type gblRow = _destination_map->getGlobalElement (lclRow);
    // Tell the sparse matrix that we are done adding entries to it.
    _crs_matrix->fillComplete ();

    std::cout << "fill comp;lete finished" << std::endl;
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
      _source->replaceGlobalValue(4+cumulative_points_per_process[_ranks(i)]+_indices(i),0,source_values(i));

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
