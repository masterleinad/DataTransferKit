/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_DistributedSearchTree.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include <boost/geometry.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

// The `out` and `success` parameters come from the Teuchos unit testing macros
// expansion.
template <typename Query, typename DeviceType>
void checkResults(
    DataTransferKit::DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> const &queries,
    std::vector<int> const &indices_ref, std::vector<int> const &offset_ref,
    std::vector<int> const &ranks_ref, bool &success,
    Teuchos::FancyOStream &out )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    tree.query( queries, indices, offset, ranks );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    deep_copy( ranks_host, ranks );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    TEST_COMPARE_ARRAYS( ranks_host, ranks_ref );
}

template <typename Query, typename DeviceType>
void checkResults(
    DataTransferKit::DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> const &queries,
    std::vector<int> const &indices_ref, std::vector<int> const &offset_ref,
    std::vector<int> const &ranks_ref, std::vector<double> const &distances_ref,
    bool &success, Teuchos::FancyOStream &out )
{
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    Kokkos::View<double *, DeviceType> distances( "distances" );
    tree.query( queries, indices, offset, ranks, distances );

    auto indices_host = Kokkos::create_mirror_view( indices );
    deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    deep_copy( offset_host, offset );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    deep_copy( ranks_host, ranks );
    auto distances_host = Kokkos::create_mirror_view( distances );
    deep_copy( distances_host, distances );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    TEST_COMPARE_ARRAYS( ranks_host, ranks_ref );
    TEST_COMPARE_FLOATING_ARRAYS( distances_host, distances_ref, 1e-14 );
}

template <typename DeviceType>
DataTransferKit::DistributedSearchTree<DeviceType>
makeDistributedSearchTree( Teuchos::RCP<const Teuchos::Comm<int>> const &comm,
                           std::vector<DataTransferKit::Box> const &b )
{
    int const n = b.size();
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    for ( int i = 0; i < n; ++i )
        boxes_host( i ) = b[i];
    Kokkos::deep_copy( boxes, boxes_host );
    return DataTransferKit::DistributedSearchTree<DeviceType>( comm, boxes );
}

// FIXME: Move into separate header file. This is copy/paste from
// tstLinearBVH.cpp
void testBoxEquality( DataTransferKit::Box const &l,
                      DataTransferKit::Box const &r, bool &success,
                      Teuchos::FancyOStream &out )
{
    TEST_EQUALITY( l[0], r[0] );
    TEST_EQUALITY( l[1], r[1] );
    TEST_EQUALITY( l[2], r[2] );
    TEST_EQUALITY( l[3], r[3] );
    TEST_EQUALITY( l[4], r[4] );
    TEST_EQUALITY( l[5], r[5] );
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType>
makeOverlapQueries( std::vector<DataTransferKit::Box> const &boxes )
{
    int const n = boxes.size();
    Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType> queries(
        "overlap_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::Details::overlap( boxes[i] );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Details::Nearest *, DeviceType>
makeNearestQueries(
    std::vector<std::pair<DataTransferKit::Point, int>> const &points )
{
    // NOTE: `points` is not a very descriptive name here. It stores both the
    // actual point and the number k of neighbors to query for.
    int const n = points.size();
    Kokkos::View<DataTransferKit::Details::Nearest *, DeviceType> queries(
        "nearest_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::Details::nearest(
            points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Details::Within *, DeviceType> makeWithinQueries(
    std::vector<std::pair<DataTransferKit::Point, double>> const &points )
{
    // NOTE: `points` is not a very descriptive name here. It stores both the
    // actual point and the radius for the search around that point.
    int const n = points.size();
    Kokkos::View<DataTransferKit::Details::Within *, DeviceType> queries(
        "within_queries", n );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < n; ++i )
        queries_host( i ) = DataTransferKit::Details::within(
            points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );
    return queries;
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DistributedSearchTree, hello_world,
                                   DeviceType )
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = Teuchos::rank( *comm );
    int const comm_size = Teuchos::size( *comm );

    DataTransferKit::DistributedSearchTreeImpl<DeviceType>::epsilon = 0.5;
    int const n = 4;
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    // [  rank 0       [  rank 1       [  rank 2       [  rank 3       [
    // x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---
    // ^   ^   ^   ^
    // 0   1   2   3   ^   ^   ^   ^
    //                 0   1   2   3   ^   ^   ^   ^
    //                                 0   1   2   3   ^   ^   ^   ^
    //                                                 0   1   2   3
    for ( int i = 0; i < n; ++i )
    {
        DataTransferKit::Box box;
        DataTransferKit::Point point = {{(double)i / n + comm_rank, 0., 0.}};
        DataTransferKit::Details::expand( box, point );
        boxes_host( i ) = box;
    }
    Kokkos::deep_copy( boxes, boxes_host );

    DataTransferKit::DistributedSearchTree<DeviceType> tree( comm, boxes );

    // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
    // |               |               |               |               |
    // |               |               |               x   x   x   x   |
    // |               |               |               |<------0------>|
    // |               |               x   x   x   x   x               |
    // |               |               |<------1------>|               |
    // |               x   x   x   x   x               |               |
    // |               |<------2------>|               |               |
    // x   x   x   x   x               |               |               |
    // |<------3------>|               |               |               |
    // |               |               |               |               |
    Kokkos::View<DataTransferKit::Details::Within *, DeviceType> queries(
        "queries", 1 );
    auto queries_host = Kokkos::create_mirror_view( queries );
    queries_host( 0 ) = DataTransferKit::Details::within(
        {{0.5 + comm_size - 1 - comm_rank, 0., 0.}}, 0.5 );
    deep_copy( queries, queries_host );

    // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
    // |               |               |               |               |
    // |               |               |           x   x   x           |
    // |               |           x   x   x        <--0-->            |
    // |           x   x   x        <--1-->            |               |
    // x   x        <--2-->            |               |               |
    // 3-->            |               |               |               |
    // |               |               |               |               |
    Kokkos::View<DataTransferKit::Details::Nearest *, DeviceType>
        nearest_queries( "nearest_queries", 1 );
    auto nearest_queries_host = Kokkos::create_mirror_view( nearest_queries );
    nearest_queries_host( 0 ) = DataTransferKit::Details::nearest(
        {{0.0 + comm_size - 1 - comm_rank, 0., 0.}},
        comm_rank < comm_size - 1 ? 3 : 2 );
    deep_copy( nearest_queries, nearest_queries_host );

    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    tree.query( queries, indices, offset, ranks );

    auto indices_host = Kokkos::create_mirror_view( indices );
    Kokkos::deep_copy( indices_host, indices );
    auto ranks_host = Kokkos::create_mirror_view( ranks );
    Kokkos::deep_copy( ranks_host, ranks );
    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( offset_host, offset );

    TEST_EQUALITY( offset_host.extent( 0 ), 2 );
    TEST_EQUALITY( offset_host( 0 ), 0 );
    TEST_EQUALITY( offset_host( 1 ), indices_host.extent_int( 0 ) );
    TEST_EQUALITY( indices_host.extent( 0 ), ranks_host.extent( 0 ) );
    TEST_EQUALITY( indices_host.extent( 0 ), comm_rank > 0 ? n + 1 : n );
    for ( int i = 0; i < n; ++i )
    {
        TEST_EQUALITY( n - 1 - i, indices_host( i ) );
        TEST_EQUALITY( comm_size - 1 - comm_rank, ranks_host( i ) );
    }
    if ( comm_rank > 0 )
    {
        TEST_EQUALITY( indices_host( n ), 0 );
        TEST_EQUALITY( ranks_host( n ), comm_size - comm_rank );
    }

    tree.query( nearest_queries, indices, offset, ranks );

    indices_host = Kokkos::create_mirror_view( indices );
    ranks_host = Kokkos::create_mirror_view( ranks );
    offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( ranks_host, ranks );
    Kokkos::deep_copy( offset_host, offset );

    TEST_COMPARE( n, >, 2 );
    TEST_EQUALITY( offset_host.extent( 0 ), 2 );
    TEST_EQUALITY( offset_host( 0 ), 0 );
    TEST_EQUALITY( offset_host( 1 ), indices_host.extent_int( 0 ) );
    TEST_EQUALITY( indices_host.extent( 0 ),
                   comm_rank < comm_size - 1 ? 3 : 2 );

    TEST_EQUALITY( indices_host( 0 ), 0 );
    TEST_EQUALITY( ranks_host( 0 ), comm_size - 1 - comm_rank );
    if ( comm_rank < comm_size - 1 )
    {
        TEST_EQUALITY( indices_host( 1 ), n - 1 );
        TEST_EQUALITY( ranks_host( 1 ), comm_size - 2 - comm_rank );
        TEST_EQUALITY( indices_host( 2 ), 1 );
        TEST_EQUALITY( ranks_host( 2 ), comm_size - 1 - comm_rank );
    }
    else
    {
        TEST_EQUALITY( indices_host( 1 ), 1 );
        TEST_EQUALITY( ranks_host( 1 ), comm_size - 1 - comm_rank );
    }
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DistributedSearchTree, empty_tree,
                                   DeviceType )
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = Teuchos::rank( *comm );
    int const comm_size = Teuchos::size( *comm );

    auto empty_tree = makeDistributedSearchTree<DeviceType>( comm, {} );

    TEST_ASSERT( empty_tree.empty() );
    TEST_EQUALITY( empty_tree.size(), 0 );

    testBoxEquality( empty_tree.bounds(), {}, success, out );

    checkResults( empty_tree, makeOverlapQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );

    checkResults( empty_tree, makeWithinQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );

    checkResults( empty_tree, makeNearestQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );

    checkResults( empty_tree, makeNearestQueries<DeviceType>( {} ), {}, {0}, {},
                  {}, success, out );

    // Only rank 0 has a couple spatial queries with a spatial predicate
    if ( comm_rank == 0 )
        checkResults( empty_tree,
                      makeOverlapQueries<DeviceType>( {
                          {},
                          {},
                      } ),
                      {}, {0, 0, 0}, {}, success, out );
    else
        checkResults( empty_tree, makeOverlapQueries<DeviceType>( {} ), {}, {0},
                      {}, success, out );

    // All ranks but rank 0 have a single query with a spatial predicate
    if ( comm_rank == 0 )
        checkResults( empty_tree, makeWithinQueries<DeviceType>( {} ), {}, {0},
                      {}, success, out );
    else
        checkResults( empty_tree,
                      makeWithinQueries<DeviceType>( {
                          {{{(double)comm_rank, 0., 0.}}, (double)comm_size},
                      } ),
                      {}, {0, 0}, {}, success, out );

    // All ranks but rank 0 have a single query with a nearest predicate
    if ( comm_rank == 0 )
        checkResults( empty_tree, makeNearestQueries<DeviceType>( {} ), {}, {0},
                      {}, success, out );
    else
        checkResults( empty_tree,
                      makeNearestQueries<DeviceType>( {
                          {{{0., 0., 0.}}, comm_rank},
                      } ),
                      {}, {0, 0}, {}, success, out );

    // All ranks have a single query with a nearest predicate (this version
    // returns distances as well)
    checkResults( empty_tree,
                  makeNearestQueries<DeviceType>( {
                      {{{0., 0., 0.}}, comm_size},
                  } ),
                  {}, {0, 0}, {}, {}, success, out );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DistributedSearchTree, unique_leaf_on_rank_0,
                                   DeviceType )
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = Teuchos::rank( *comm );
    int const comm_size = Teuchos::size( *comm );

    // tree has one unique leaf that lives on rank 0
    auto tree =
        ( comm_rank == 0 ? makeDistributedSearchTree<DeviceType>(
                               comm,
                               {
                                   {{0., 1., 0., 1., 0., 1.}},
                               } )
                         : makeDistributedSearchTree<DeviceType>( comm, {} ) );

    TEST_ASSERT( !tree.empty() );
    TEST_EQUALITY( tree.size(), 1 );

    testBoxEquality( tree.bounds(), {{0., 1., 0., 1., 0., 1.}}, success, out );

    checkResults( tree, makeOverlapQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );

    checkResults( tree, makeWithinQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );

    checkResults( tree, makeNearestQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );

    checkResults( tree, makeNearestQueries<DeviceType>( {} ), {}, {0}, {}, {},
                  success, out );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DistributedSearchTree, one_leaf_per_rank,
                                   DeviceType )
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int const comm_rank = Teuchos::rank( *comm );
    int const comm_size = Teuchos::size( *comm );

    // tree has one leaf per rank
    auto tree = makeDistributedSearchTree<DeviceType>(
        comm, {
                  {{(double)comm_rank, (double)comm_rank + 1., 0., 1., 0., 1.}},
              } );

    TEST_ASSERT( !tree.empty() );
    TEST_EQUALITY( (int)tree.size(), comm_size );

    testBoxEquality( tree.bounds(), {{0., (double)comm_size, 0., 1., 0., 1.}},
                     success, out );

    checkResults(
        tree,
        makeOverlapQueries<DeviceType>( {
            {{(double)comm_size - (double)comm_rank - .5,
              (double)comm_size - (double)comm_rank - .5, .5, .5, .5, .5}},
            {{(double)comm_rank + .5, (double)comm_rank + .5, .5, .5, .5, .5}},
        } ),
        {0, 0}, {0, 1, 2}, {comm_size - 1 - comm_rank, comm_rank}, success,
        out );

    checkResults( tree, makeNearestQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );
    checkResults( tree, makeOverlapQueries<DeviceType>( {} ), {}, {0}, {},
                  success, out );
}

std::vector<std::array<double, 3>>
make_random_cloud( double const Lx, double const Ly, double const Lz,
                   int const n, double const seed )
{
    std::vector<std::array<double, 3>> cloud( n );
    std::default_random_engine generator( seed );
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );
    for ( int i = 0; i < n; ++i )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        cloud[i] = {{x, y, z}};
    }
    return cloud;
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DistributedSearchTree, boost_comparison,
                                   DeviceType )
{
    namespace details = DataTransferKit::Details;
    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;
    using BPoint = bg::model::point<double, 3, bg::cs::cartesian>;
    using RTree = bgi::rtree<std::pair<BPoint, int>, bgi::linear<16>>;

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    unsigned int const comm_rank = Teuchos::rank( *comm );
    unsigned int const comm_size = Teuchos::size( *comm );

    // Construct a random cloud of point. We use the same seed on all the
    // processors.
    double const Lx = 10.0;
    double const Ly = 10.0;
    double const Lz = 10.0;
    int const n = 100;
    auto cloud = make_random_cloud( Lx, Ly, Lz, n, 0 );
    auto queries = make_random_cloud( Lx, Ly, Lz, n, 1234 );

    // Create a R-tree to compare radius search results against
    RTree rtree;
    for ( int i = 0; i < n; ++i )
    {
        auto const &point = cloud[i];
        double const x = std::get<0>( point );
        double const y = std::get<1>( point );
        double const z = std::get<2>( point );
        rtree.insert( std::make_pair( BPoint( x, y, z ), i ) );
    }

    unsigned int const local_n = n / comm_size;
    Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes(
        "bounding_boxes", local_n );
    auto bounding_boxes_host = Kokkos::create_mirror_view( bounding_boxes );
    for ( unsigned int i = 0; i < n; ++i )
    {
        if ( i % comm_size == comm_rank )
        {
            auto const &point = cloud[i];
            double const x = std::get<0>( point );
            double const y = std::get<1>( point );
            double const z = std::get<2>( point );
            bounding_boxes_host[i / comm_size] = {x, x, y, y, z, z};
        }
    }

    std::map<std::pair<unsigned int, unsigned int>, unsigned int> indices_map;
    for ( unsigned int i = 0; i < n; ++i )
        for ( unsigned int j = 0; j < comm_size; ++j )
            if ( i % comm_size == j )
                indices_map[std::make_pair( i / comm_size, j )] = i;

    Kokkos::deep_copy( bounding_boxes, bounding_boxes_host );

    // Initialize the distributed search tree
    DataTransferKit::DistributedSearchTree<DeviceType> distributed_tree(
        comm, bounding_boxes );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<double * [3], ExecutionSpace> point_coords( "point_coords",
                                                             local_n );
    auto point_coords_host = Kokkos::create_mirror_view( point_coords );
    Kokkos::View<double *, ExecutionSpace> radii( "radii", local_n );
    auto radii_host = Kokkos::create_mirror_view( radii );
    Kokkos::View<int * [2], ExecutionSpace> within_n_pts( "within_n_pts",
                                                          local_n );
    Kokkos::View<int *, ExecutionSpace> k( "distribution_k", local_n );
    auto k_host = Kokkos::create_mirror_view( k );
    std::vector<std::vector<std::pair<BPoint, int>>> returned_values_within(
        local_n );
    std::default_random_engine generator( 0 );
    std::uniform_real_distribution<double> distribution_radius(
        0.0, std::sqrt( Lx * Lx + Ly * Ly + Lz * Lz ) );
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor( sqrt( n * n ) ) );
    for ( unsigned int i = 0; i < n; ++i )
    {
        if ( i % comm_size == comm_rank )
        {
            auto const &point = queries[i];
            unsigned int const j = i / comm_size;
            double const x = std::get<0>( point );
            double const y = std::get<1>( point );
            double const z = std::get<2>( point );
            BPoint centroid( x, y, z );
            radii_host[j] = distribution_radius( generator );
            double radius = radii_host[j];

            point_coords_host( j, 0 ) = x;
            point_coords_host( j, 1 ) = y;
            point_coords_host( j, 2 ) = z;

            // use the R-tree to obtain a reference solution
            rtree.query(
                bgi::satisfies(
                    [centroid, radius]( std::pair<BPoint, int> const &val ) {
                        return bg::distance( centroid, val.first ) <= radius;
                    } ),
                std::back_inserter( returned_values_within[j] ) );
        }
    }

    Kokkos::deep_copy( point_coords, point_coords_host );
    Kokkos::deep_copy( radii, radii_host );

    Kokkos::View<details::Within *, DeviceType> within_queries(
        "within_queries", local_n );
    Kokkos::parallel_for( "register_within_queries",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, local_n ),
                          KOKKOS_LAMBDA( int i ) {
                              within_queries( i ) = details::within(
                                  {{point_coords( i, 0 ), point_coords( i, 1 ),
                                    point_coords( i, 2 )}},
                                  radii( i ) );
                          } );
    Kokkos::fence();

    // Perform the search
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> ranks( "ranks" );
    distributed_tree.query( within_queries, indices, offset, ranks );

    auto indices_host = Kokkos::create_mirror_view( indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    auto ranks_host = Kokkos::create_mirror_view( ranks );

    for ( unsigned int i = 0; i < n; ++i )
    {
        if ( i % comm_size == comm_rank )
        {
            unsigned int k = i / comm_size;
            auto const &ref = returned_values_within[k];
            std::set<int> ref_ids;
            for ( auto const &id : ref )
                ref_ids.emplace( id.second );
            for ( int j = offset_host( k ); j < offset_host( k + 1 ); ++j )
            {
                TEST_ASSERT( ref_ids.count( indices_map[std::make_pair(
                                 indices_host( j ), ranks( j ) )] ) != 0 );
            }
        }
    }
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DistributedSearchTree, hello_world,  \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DistributedSearchTree, empty_tree,   \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(                                      \
        DistributedSearchTree, unique_leaf_on_rank_0, DeviceType##NODE )       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(                                      \
        DistributedSearchTree, one_leaf_per_rank, DeviceType##NODE )           \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DistributedSearchTree,               \
                                          boost_comparison, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
