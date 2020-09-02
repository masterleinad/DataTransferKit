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

#ifndef DTK_SPLINE_OPERATOR_DECL_HPP
#define DTK_SPLINE_OPERATOR_DECL_HPP

#include <DTK_CompactlySupportedRadialBasisFunctions.hpp>
#include <DTK_MultivariatePolynomialBasis.hpp>
#include <DTK_PointCloudOperator.hpp>

#include <Tpetra_CrsMatrix.hpp>

#include <mpi.h>

namespace DataTransferKit
{

/**
 * This class implements a function reconstruction technique for arbitrary point
 * cloud based on a spline discretization. In this method, support
 * and subsequently the data transfer operator is constructed through solutions
 * to local least square kernels defined by compactly supported radial basis
 * functions.
 *
 * The class is templated on the DeviceType, the radial basis function
 * (Wendland<0>, Wendland<2>, Wendland<4>, Wendland<6>, Wu<2>, Wu<4>,
 * Buhmann<2>, Buhmann<3>, or Buhmann<4>) and polynonial basis (<Constant, DIM>,
 * <Linear, DIM>, or <Quadratic, DIM>).
 */
template <typename DeviceType,
          typename CompactlySupportedRadialBasisFunction = Wendland<0>,
          typename PolynomialBasis = MultivariatePolynomialBasis<Linear, 3>>
class SplineOperator : public PointCloudOperator<DeviceType>
{
    using ExecutionSpace = typename DeviceType::execution_space;
     using VectorType = Tpetra::MultiVector<>;
    using OperatorType = Tpetra::Operator<>;
    using ScalarType = double;

  public:
    SplineOperator(
        MPI_Comm comm,
        Kokkos::View<Coordinate const **, DeviceType> source_points,
        Kokkos::View<Coordinate const **, DeviceType> target_points );

    void
    apply( Kokkos::View<double const *, DeviceType> source_values,
           Kokkos::View<double *, DeviceType> target_values ) const override;

  private:
    MPI_Comm _comm;
    unsigned int const _n_source_points;
    Kokkos::View<int *, DeviceType> _offset;
    Kokkos::View<int *, DeviceType> _ranks;
    Kokkos::View<int *, DeviceType> _indices;

    Kokkos::View<int *, DeviceType> target_offset;
    Kokkos::View<int *, DeviceType> target_ranks;
    Kokkos::View<int *, DeviceType> target_indices;

    Kokkos::View<double *, DeviceType> _coeffs;
    Teuchos::RCP<Tpetra::CrsMatrix<>> _crs_matrix;
    Teuchos::RCP<Tpetra::Map<>> _destination_map;
    Teuchos::RCP<Tpetra::Map<>> _source_map;
    std::vector<int> cumulative_points_per_process;
    Teuchos::RCP<Tpetra::MultiVector<>> _source;
    Teuchos::RCP<Tpetra::MultiVector<>> _destination;

       // Prolongation operator.
    Teuchos::RCP<const Tpetra::Operator<double,int,GlobalOrdinal>> S;

    // Coefficient matrix polynomial component.
    Teuchos::RCP<const Tpetra::Operator<double,int,GlobalOrdinal>> P;

    // Coefficient matrix basis component.
    Teuchos::RCP<const Tpetra::Operator<double,int,GlobalOrdinal>> M;

    // Evaluation matrix polynomial component.
    Teuchos::RCP<const Tpetra::Operator<double,int,GlobalOrdinal>> Q;

    // Evaluation matrix basis component.
    Teuchos::RCP<const Tpetra::Operator<double,int,GlobalOrdinal>> N;
};

} // end namespace DataTransferKit

#endif
