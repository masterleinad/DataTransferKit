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

    // For each target point, query the n_neighbors points closest to the
    // target.
    auto queries =
        Details::SplineOperatorImpl<DeviceType>::makeKNNQueries(
            target_points, PolynomialBasis::size );

    // Perform the actual search.
    search_tree.query( queries, _indices, _offset, _ranks );

    // Retrieve the coordinates of all source points that met the predicates.
    // NOTE: This is the last collective.
    source_points = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, _ranks, _indices, source_points );

    int local_cumulative_points = 0;
    MPI_Scan(&_n_source_points, &local_cumulative_points, 1, MPI_INT, MPI_SUM, comm);

    int n_processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    std::vector<int> cumulative_points_per_process(n_processes+1);
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
    Tpetra::Map<> source_map(cumulative_points_per_process.back(), kokkos_indices, 0, teuchos_comm);  

    // Transform source points
    source_points = Details::SplineOperatorImpl<
        DeviceType>::transformSourceCoordinates( source_points, _offset,
                                                 target_points );
    target_points = Kokkos::View<Coordinate **, DeviceType>( "empty", 0, 0 );

    // Build P (vandermonde matrix)
    // P is a single 1D storage for multiple P_i matrices. Each matrix is of
    // size (#source_points_for_specific_target_point, basis_size)
    auto p =
        Details::SplineOperatorImpl<DeviceType>::computeVandermonde(
            source_points, PolynomialBasis() );

    // To build the radial basis function, we need to define the radius of the
    // radial basis function. Since we use kNN, we need to compute the radius.
    // We only need the coordinates of the source points because of the
    // transformation of the coordinates.
    auto radius =
        Details::SplineOperatorImpl<DeviceType>::computeRadius(
            source_points, _offset );

    // Build phi (weight matrix)
    auto phi =
        Details::SplineOperatorImpl<DeviceType>::computeWeights(
            source_points, radius, CompactlySupportedRadialBasisFunction() );

    // Build A (moment matrix)
    auto a =
        Details::SplineOperatorImpl<DeviceType>::computeMoments(
            _offset, p, phi );

    // TODO: it is computationally unnecessary to compute the pseudo-inverse as
    // MxM (U*E^+*V) as it will later be just used to do MxV. We could instead
    // return the (U,E^+,V) and do the MxV multiplication. But for now, it's OK.
    auto t = Details::SplineOperatorImpl<DeviceType>::invertMoments(
        a, PolynomialBasis::size );
    auto inv_a = std::get<0>( t );

    // std::get<1>(t) returns the number of undetermined system. However, this
    // is not enough to know if we will lose order of accuracy. For example, if
    // all the points are aligned, the system will be underdetermined. However
    // this is not a problem if we found at least three points since this is
    // enough to define a quadratic function. Therefore, not only we need to
    // know the rank deficiency but also the dimension of the problem.

    // NOTE: This assumes that the polynomial basis evaluated at {0,0,0} is
    // going to be [1, 0, 0, ..., 0]^T.
    _coeffs = Details::SplineOperatorImpl<
        DeviceType>::computePolynomialCoefficients( _offset, inv_a, p, phi,
                                                    PolynomialBasis::size );


    // build matrix
    int n_global_points;
    MPI_Allreduce(&_n_source_points, &n_global_points, 1, MPI_INT, MPI_SUM, comm);
    constexpr global_ordinal_type indexBase = 0;
    _map = Teuchos::rcp (new Tpetra::Map<> (n_global_points, indexBase, teuchos_comm));

    _crs_matrix = Teuchos::rcp(new Tpetra::CrsMatrix<>(_map, 3));
    // Fill the sparse matrix, one row at a time.
    const scalar_type two = static_cast<scalar_type> (2.0);
    const scalar_type negOne = static_cast<scalar_type> (-1.0);
    for (local_ordinal_type lclRow = 0; lclRow < static_cast<local_ordinal_type> (_n_source_points); ++lclRow) 
    {
      const global_ordinal_type gblRow = _map->getGlobalElement (lclRow);
      // _crs_matrix(0, 0:1) = [2, -1]
      if (gblRow == 0) 
      {
        _crs_matrix->insertGlobalValues (gblRow, Teuchos::tuple<global_ordinal_type> (gblRow, gblRow + 1), Teuchos::tuple<scalar_type> (two, negOne));
      }
      // _crs_matrix(N-1, N-2:N-1) = [-1, 2]
      else if (static_cast<int> (gblRow) == n_global_points - 1) 
      {
        _crs_matrix->insertGlobalValues (gblRow, Teuchos::tuple<global_ordinal_type> (gblRow - 1, gblRow), Teuchos::tuple<scalar_type> (negOne, two));
      }
      // _crs_matrix(i, i-1:i+1) = [-1, 2, -1]
      else 
      {
        _crs_matrix->insertGlobalValues (gblRow, Teuchos::tuple<global_ordinal_type> (gblRow - 1, gblRow, gblRow + 1), Teuchos::tuple<scalar_type> (negOne, two, negOne));
      }
    }
    // Tell the sparse matrix that we are done adding entries to it.
    _crs_matrix->fillComplete ();
}

template <typename DeviceType, typename CompactlySupportedRadialBasisFunction,
          typename PolynomialBasis>
void SplineOperator<
    DeviceType, CompactlySupportedRadialBasisFunction, PolynomialBasis>::
    apply( Kokkos::View<double const *, DeviceType> source_values,
           Kokkos::View<double *, DeviceType> target_values ) const
{
    // Precondition: check that the source and the target are properly sized
    DTK_REQUIRE( source_values.extent( 0 ) == _n_source_points );
    DTK_REQUIRE( target_values.extent( 0 ) == _offset.extent( 0 ) - 1 );

    // Retrieve values for all source points
    source_values = Details::NearestNeighborOperatorImpl<DeviceType>::fetch(
        _comm, _ranks, _indices, source_values );

    using VectorType = Tpetra::MultiVector<>;
    using OperatorType = Tpetra::Operator<>;
    using ScalarType = double;

    auto source = Teuchos::rcp( new VectorType(_map, 1));
    auto destination = Teuchos::rcp( new VectorType(_map, 1));

    // copy source_values to source

    auto problem = Teuchos::rcp (new Belos::LinearProblem<ScalarType,VectorType,OperatorType>(_crs_matrix, source, destination));
    problem->setProblem();

    Teuchos::RCP<Teuchos::ParameterList> params;
    // params->set(...);
    Belos::BlockGmresSolMgr<ScalarType,VectorType,OperatorType> solver( problem, params );
    auto ret = solver.solve();
    (void) ret;
    DTK_REQUIRE(ret == Belos::Converged);
    auto solution = problem->getLHS();

    // copy solution to target_values
}

} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_SPLINE_OPERATOR_INSTANT( NODE )                      \
    template class SplineOperator<typename NODE::device_type>;     \
    template class SplineOperator<                                 \
        typename NODE::device_type, Wendland<0>,                               \
        MultivariatePolynomialBasis<Quadratic, 3>>;

#endif
