/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "../test/DTK_BoostRTreeHelpers.hpp"
#include "../test/DTK_NanoflannAdapters.hpp"

#include <DTK_LinearBVH.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

#include <benchmark/benchmark.h>

#include <point_clouds.hpp>

#include <mpi.h>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>

#if defined( HAVE_DTK_BOOST ) && defined( KOKKOS_ENABLE_SERIAL )
class BoostRTree
{
  public:
    using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    using device_type = DeviceType;

    BoostRTree( Kokkos::View<DataTransferKit::Point *, DeviceType> points )
    {
        _tree = BoostRTreeHelpers::makeRTree( points );
    }

    template <typename Query>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset )
    {
        std::tie( offset, indices ) =
            BoostRTreeHelpers::performQueries( _tree, queries );
    }

    template <typename Query>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset, int )
    {
        std::tie( offset, indices ) =
            BoostRTreeHelpers::performQueries( _tree, queries );
    }

  private:
    BoostRTreeHelpers::RTree<DataTransferKit::Point> _tree;
};
#endif

#ifdef KOKKOS_ENABLE_SERIAL
class NanoflannKDTree
{
  public:
    using DatasetAdapter =
        DataTransferKit::NanoflannPointCloudAdapter<Kokkos::HostSpace>;
    using DistanceType = nanoflann::L2_Simple_Adaptor<double, DatasetAdapter>;
    using KDTree =
        nanoflann::KDTreeSingleIndexAdaptor<DistanceType, DatasetAdapter, 3,
                                            size_t>;
    using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    using device_type = DeviceType;

    NanoflannKDTree( Kokkos::View<DataTransferKit::Point *, DeviceType> points )
        : _dataset_adapter( points )
        , _tree( 3, _dataset_adapter )
    {
        _tree.buildIndex();
    }

    template <typename Query>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset )
    {
        unsigned int const n_queries = queries.extent( 0 );
        Kokkos::realloc( offset, n_queries + 1 );
        std::vector<std::pair<size_t, double>> returned_indices_distances;
        for ( unsigned int i = 0; i < n_queries; ++i )
        {
            size_t const k = queries( i )._k;
            double const *const query_point = queries( i )._geometry._coords;
            std::vector<size_t> query_indices( k );
            std::vector<double> distances_sq( k );
            offset( i ) = _tree.knnSearch( query_point, k, query_indices.data(),
                                           distances_sq.data() );
            for ( int j = 0; j < offset( j ); ++j )
                returned_indices_distances.push_back( std::make_pair(
                    query_indices[j], std::sqrt( distances_sq[j] ) ) );
        }
        DataTransferKit::exclusivePrefixSum( offset );
        unsigned int const n_results = DataTransferKit::lastElement( offset );
        Kokkos::realloc( indices, n_results );
        for ( unsigned int i = 0; i < n_queries; ++i )
            for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                indices( j ) = returned_indices_distances[j].first;
    }

    template <typename Query>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset, int )
    {
        unsigned int const n_queries = queries.extent( 0 );
        Kokkos::realloc( offset, n_queries + 1 );
        std::vector<std::pair<size_t, double>> returned_indices_distances;
        for ( unsigned int i = 0; i < n_queries; ++i )
        {
            auto const sphere = queries( i )._geometry;
            double const radius = sphere.radius();
            double const *const centroid = sphere.centroid()._coords;
            std::vector<std::pair<size_t, double>> ret_matches;
            nanoflann::SearchParams params;
            offset( i ) = _tree.radiusSearch( centroid, radius * radius,
                                              ret_matches, params );
            returned_indices_distances.insert( returned_indices_distances.end(),
                                               ret_matches.begin(),
                                               ret_matches.end() );
        }
        DataTransferKit::exclusivePrefixSum( offset );
        unsigned int const n_results = DataTransferKit::lastElement( offset );
        Kokkos::realloc( indices, n_results );
        for ( unsigned int i = 0; i < n_queries; ++i )
            for ( int j = offset( i ); j < offset( i + 1 ); ++j )
                indices( j ) = returned_indices_distances[j].first;
    }

  private:
    DatasetAdapter _dataset_adapter;
    KDTree _tree;
};
#endif

template <typename DeviceType>
Kokkos::View<DataTransferKit::Point *, DeviceType>
constructPoints( int n_values, PointCloudType point_cloud_type )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        Kokkos::ViewAllocateWithoutInitializing( "random_points" ), n_values );
    // Generate random points uniformely distributed within a box.  The edge
    // length of the box chosen such that object density (here objects will be
    // boxes 2x2x2 centered around a random point) will remain constant as
    // problem size is changed.
    auto const a = std::cbrt( n_values );
    generatePointCloud( point_cloud_type, a, random_points );

    return random_points;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
makeNearestQueries( int n_values, int n_queries, int n_neighbors,
                    PointCloudType target_point_cloud_type )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        Kokkos::ViewAllocateWithoutInitializing( "random_points" ), n_queries );
    auto const a = std::cbrt( n_values );
    generatePointCloud( target_point_cloud_type, a, random_points );

    Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
        queries( Kokkos::ViewAllocateWithoutInitializing( "queries" ),
                 n_queries );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for(
        "bvh_driver:setup_knn_search_queries",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
        KOKKOS_LAMBDA( int i ) {
            queries( i ) = DataTransferKit::nearest<DataTransferKit::Point>(
                random_points( i ), n_neighbors );
        } );
    Kokkos::fence();
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Within *, DeviceType>
makeSpatialQueries( int n_values, int n_queries, int n_neighbors,
                    PointCloudType target_point_cloud_type )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        Kokkos::ViewAllocateWithoutInitializing( "random_points" ), n_queries );
    auto const a = std::cbrt( n_values );
    generatePointCloud( target_point_cloud_type, a, random_points );

    Kokkos::View<DataTransferKit::Within *, DeviceType> queries(
        Kokkos::ViewAllocateWithoutInitializing( "queries" ), n_queries );
    // radius chosen in order to control the number of results per query
    double const r = 2. * std::cbrt( static_cast<double>( n_neighbors ) * 3. /
                                     ( 4. * M_PI ) );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for( "bvh_driver:setup_radius_search_queries",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int i ) {
                              queries( i ) = DataTransferKit::within(
                                  random_points( i ), r );
                          } );
    Kokkos::fence();
    return queries;
}

template <class TreeType>
void BM_construction( benchmark::State &state )
{
    using DeviceType = typename TreeType::device_type;
    int const n_values = state.range( 0 );
    PointCloudType point_cloud_type =
        static_cast<PointCloudType>( state.range( 1 ) );
    auto points = constructPoints<DeviceType>( n_values, point_cloud_type );

    for ( auto _ : state )
    {
        auto const start = std::chrono::high_resolution_clock::now();
        TreeType index( points );
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        state.SetIterationTime( elapsed_seconds.count() );
    }
}

template <class TreeType>
void BM_knn_search( benchmark::State &state )
{
    using DeviceType = typename TreeType::device_type;
    int const n_values = state.range( 0 );
    int const n_queries = state.range( 1 );
    int const n_neighbors = state.range( 2 );
    PointCloudType const source_point_cloud_type =
        static_cast<PointCloudType>( state.range( 3 ) );
    PointCloudType const target_point_cloud_type =
        static_cast<PointCloudType>( state.range( 4 ) );

    TreeType index(
        constructPoints<DeviceType>( n_values, source_point_cloud_type ) );
    auto const queries = makeNearestQueries<DeviceType>(
        n_values, n_queries, n_neighbors, target_point_cloud_type );

    for ( auto _ : state )
    {
        Kokkos::View<int *, DeviceType> offset( "offset" );
        Kokkos::View<int *, DeviceType> indices( "indices" );
        auto const start = std::chrono::high_resolution_clock::now();
        index.query( queries, indices, offset );
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        state.SetIterationTime( elapsed_seconds.count() );
    }
}

template <class TreeType>
void BM_radius_search( benchmark::State &state )
{
    using DeviceType = typename TreeType::device_type;
    int const n_values = state.range( 0 );
    int const n_queries = state.range( 1 );
    int const n_neighbors = state.range( 2 );
    int const buffer_size = state.range( 3 );
    PointCloudType const source_point_cloud_type =
        static_cast<PointCloudType>( state.range( 4 ) );
    PointCloudType const target_point_cloud_type =
        static_cast<PointCloudType>( state.range( 5 ) );

    TreeType index(
        constructPoints<DeviceType>( n_values, source_point_cloud_type ) );
    auto const queries = makeSpatialQueries<DeviceType>(
        n_values, n_queries, n_neighbors, target_point_cloud_type );

    bool first_pass = true;
    for ( auto _ : state )
    {
        Kokkos::View<int *, DeviceType> offset( "offset" );
        Kokkos::View<int *, DeviceType> indices( "indices" );
        auto const start = std::chrono::high_resolution_clock::now();
        index.query( queries, indices, offset, buffer_size );
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        state.SetIterationTime( elapsed_seconds.count() );

        if ( first_pass )
        {
            auto offset_clone = DataTransferKit::clone( offset );
            DataTransferKit::adjacentDifference( offset, offset_clone );
            double const max = DataTransferKit::max( offset_clone );
            double const avg =
                DataTransferKit::lastElement( offset ) / n_queries;
            auto offset_clone_subview = Kokkos::subview(
                offset_clone,
                std::make_pair( 1, offset_clone.extent_int( 0 ) ) );
            double const min = DataTransferKit::min( offset_clone_subview );

            std::ostream &os = std::cout;
            os << "min number of neighbors " << min << "\n";
            os << "max number of neighbors " << max << "\n";
            os << "avg number of neighbors " << avg << "\n";

            first_pass = false;
        }
    }
}

class KokkosScopeGuard
{
  public:
    KokkosScopeGuard( int &argc, char *argv[] )
    {
        Kokkos::initialize( argc, argv );
    }
    ~KokkosScopeGuard() { Kokkos::finalize(); }
};

#define REGISTER_BENCHMARK( TreeType )                                         \
    BENCHMARK_TEMPLATE( BM_construction, TreeType )                            \
        ->Args( {n_values, source_point_cloud_type} )                          \
        ->UseManualTime()                                                      \
        ->Unit( benchmark::kMicrosecond );                                     \
    BENCHMARK_TEMPLATE( BM_knn_search, TreeType )                              \
        ->Args( {n_values, n_queries, n_neighbors, source_point_cloud_type,    \
                 target_point_cloud_type} )                                    \
        ->UseManualTime()                                                      \
        ->Unit( benchmark::kMicrosecond );                                     \
    BENCHMARK_TEMPLATE( BM_radius_search, TreeType )                           \
        ->Args( {n_values, n_queries, n_neighbors, buffer_size,                \
                 source_point_cloud_type, target_point_cloud_type} )           \
        ->UseManualTime()                                                      \
        ->Unit( benchmark::kMicrosecond );

int main( int argc, char *argv[] )
{
    // This is necessary on summit and ascent
    int required = MPI_THREAD_SERIALIZED;
    int provided;
    MPI_Init_thread( &argc, &argv, required, &provided );

    KokkosScopeGuard guard( argc, argv );

    bool const throw_exceptions = false;
    bool const recognise_all_options = false;
    Teuchos::CommandLineProcessor clp( throw_exceptions,
                                       recognise_all_options );
    int n_values = 50000;
    int n_queries = 20000;
    int n_neighbors = 10;
    int buffer_size = 0;
    std::string source_pt_cloud = "filled_box";
    std::string target_pt_cloud = "filled_box";
    clp.setOption( "values", &n_values, "number of indexable values (source)" );
    clp.setOption( "queries", &n_queries, "number of queries (target)" );
    clp.setOption( "neighbors", &n_neighbors,
                   "desired number of results per query" );
    clp.setOption( "buffer", &buffer_size,
                   "size for buffer optimization in radius search" );
    clp.setOption( "source-point-cloud-type", &source_pt_cloud,
                   "shape of the source point cloud" );
    clp.setOption( "target-point-cloud-type", &target_pt_cloud,
                   "shape of the target point cloud" );

    // Google benchmark only supports integer arguments (see
    // https://github.com/google/benchmark/issues/387), so we map the string to
    // an enum.
    std::map<std::string, PointCloudType> to_point_cloud_enum;
    to_point_cloud_enum["filled_box"] = PointCloudType::filled_box;
    to_point_cloud_enum["hollow_box"] = PointCloudType::hollow_box;
    to_point_cloud_enum["filled_sphere"] = PointCloudType::filled_sphere;
    to_point_cloud_enum["hollow_sphere"] = PointCloudType::hollow_sphere;
    int source_point_cloud_type = to_point_cloud_enum.at( source_pt_cloud );
    int target_point_cloud_type = to_point_cloud_enum.at( target_pt_cloud );

    switch ( clp.parse( argc, argv, NULL ) )
    {
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
        return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        clp.printHelpMessage( "benchmark", std::cout );
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

    // benchmark::Initialize() calls exit(0) when `--help` so register
    // Kokkos::finalize() to be called on normal program termination.
    std::atexit( Kokkos::finalize );
    benchmark::Initialize( &argc, argv );

    // Throw if an option is not recognised
    clp.throwExceptions( true );
    clp.recogniseAllOptions( true );
    switch ( clp.parse( argc, argv, NULL ) )
    {
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
        return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

    namespace dtk = DataTransferKit;

#ifdef KOKKOS_ENABLE_SERIAL
    using Serial = Kokkos::Compat::KokkosSerialWrapperNode::device_type;
    REGISTER_BENCHMARK( dtk::BVH<Serial> );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMP = Kokkos::Compat::KokkosOpenMPWrapperNode::device_type;
    REGISTER_BENCHMARK( dtk::BVH<OpenMP> );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using Cuda = Kokkos::Compat::KokkosCudaWrapperNode::device_type;
    REGISTER_BENCHMARK( dtk::BVH<Cuda> );
#endif

#if defined( HAVE_DTK_BOOST ) && defined( KOKKOS_ENABLE_SERIAL )
    REGISTER_BENCHMARK( BoostRTree );
#endif

#ifdef KOKKOS_ENABLE_SERIAL
    REGISTER_BENCHMARK( NanoflannKDTree );
#endif

    benchmark::RunSpecifiedBenchmarks();

    MPI_Finalize();

    return EXIT_SUCCESS;
}
