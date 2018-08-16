/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_Point.hpp>

#include <Kokkos_Core.hpp>

#include <nanoflann.hpp>

#include <vector>

#include <Teuchos_UnitTestHarness.hpp>

#include "DTK_NanoflannAdapters.hpp"

TEUCHOS_UNIT_TEST( NanoflannAdapters, PointCloud )
{
    Kokkos::View<DataTransferKit::Point[2], Kokkos::HostSpace> v( "v" );
    v( 0 ) = {{0., 0., 0.}};
    v( 1 ) = {{1., 1., 1.}};

    using DatasetAdapter =
        DataTransferKit::NanoflannPointCloudAdapter<Kokkos::HostSpace>;
    using DistanceType = nanoflann::L2_Simple_Adaptor<double, DatasetAdapter>;
    using KDTree =
        nanoflann::KDTreeSingleIndexAdaptor<DistanceType, DatasetAdapter, 3,
                                            size_t>;

    DatasetAdapter dataset_adapter( v );
    KDTree kdtree( 3, dataset_adapter );
    kdtree.buildIndex();

    size_t const k = 3;
    std::vector<size_t> indices( k );
    std::vector<double> distances_sq( k );
    std::array<double, 3> query_point = {{1., 0., 0.}};
    size_t const n = kdtree.knnSearch( query_point.data(), k, indices.data(),
                                       distances_sq.data() );
    TEST_EQUALITY( n, 2 );
    TEST_EQUALITY( indices[0], 0 );
    TEST_FLOATING_EQUALITY( distances_sq[0], 1., 1e-14 );
    TEST_EQUALITY( indices[1], 1 );
    TEST_FLOATING_EQUALITY( distances_sq[1], 2., 1e-14 );

    std::cout << "knnSearch\n";
    for ( size_t i = 0; i < n; ++i )
        std::cout << indices[i] << "  " << distances_sq[i] << "\n";

    std::vector<std::pair<size_t, double>> indices_distances;
    double const radius = 1.1;
    nanoflann::SearchParams search_params;
    size_t const m = kdtree.radiusSearch( query_point.data(), radius,
                                          indices_distances, search_params );
    TEST_EQUALITY( m, 1 );
    TEST_EQUALITY( indices_distances[0].first, 0 );
    TEST_FLOATING_EQUALITY( indices_distances[0].second, 1., 1e-14 );

    std::cout << "radiusSearch\n";
    for ( size_t j = 0; j < m; ++j )
        std::cout << indices_distances[j].first << "  "
                  << indices_distances[j].second << "\n";
}
