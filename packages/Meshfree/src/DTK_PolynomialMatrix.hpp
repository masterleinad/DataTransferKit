//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
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
 * \file   DTK_PolynomialMatrix.hpp
 * \author Stuart R. Slattery
 * \brief  Polynomial matrix.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_POLYNOMIALMATRIX_HPP
#define DTK_POLYNOMIALMATRIX_HPP

#include <DTK_Types.h>

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

#include "DTK_DBC.hpp"
#include "DTK_PolynomialMatrix.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_OpaqueWrapper.hpp>

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#include <mpi.h>
#endif

#include <Tpetra_Export.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \class PolynomialMatrix
 * \brief Vector apply implementation for polynomial matrices.
 */
//---------------------------------------------------------------------------//
template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal,
          typename Node>
class PolynomialMatrix
    : public Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>
{
    using DeviceType = typename Node::device_type;
    using Map = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using MultiVector =
        Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  public:
    PolynomialMatrix( Kokkos::View<double **, DeviceType> vandermonde,
                      const Teuchos::RCP<const Map> &domain_map,
                      const Teuchos::RCP<const Map> &range_map )
        : _vandermonde( vandermonde )
        , _domain_map( domain_map )
        , _range_map( range_map )
    {
    }

    Teuchos::RCP<const Map> getDomainMap() const override
    {
        return _domain_map;
    }

    Teuchos::RCP<const Map> getRangeMap() const override { return _range_map; }

    void
    apply( const MultiVector &X, MultiVector &Y,
           Teuchos::ETransp mode = Teuchos::NO_TRANS,
           Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
           Scalar beta = Teuchos::ScalarTraits<Scalar>::zero() ) const override
    {
        DTK_REQUIRE( _domain_map->isSameAs( *( X.getMap() ) ) );
        DTK_REQUIRE( _range_map->isSameAs( *( Y.getMap() ) ) );
        DTK_REQUIRE( X.getNumVectors() == Y.getNumVectors() );

        using ExecutionSpace = typename DeviceType::execution_space;

        auto comm = _domain_map->getComm();
#ifdef HAVE_MPI
        Teuchos::RCP<const Teuchos::MpiComm<int>> mpi_comm =
            Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>( comm );
        Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm>> opaque_comm =
            mpi_comm->getRawMpiComm();
        MPI_Comm raw_comm = ( *opaque_comm )();
#endif

        // Get the size of the problem and view of the local vectors.
        int const local_length = _vandermonde.extent( 0 );
        int const poly_size = _vandermonde.extent( 1 );
        int const num_vec = X.getNumVectors();

        // To avoid capturing *this
        auto vandermonde = _vandermonde;

        Y.scale( beta );

        if ( mode == Teuchos::NO_TRANS )
        {
            Kokkos::View<double **, DeviceType> x_poly( "x_poly", poly_size,
                                                        num_vec );
            if ( 0 == comm()->getRank() )
            {
                auto x_view = X.getLocalViewDevice();
                auto const n = x_view.extent( 0 );
                Kokkos::deep_copy(
                    x_poly, Kokkos::subview(
                                x_view, Kokkos::make_pair( n - poly_size, n ),
                                Kokkos::ALL ) );
            }

#ifdef HAVE_MPI
            // Broadcast the polynomial components of X from the root rank.
            auto x_poly_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, x_poly );
            MPI_Bcast( x_poly_host.data(), poly_size * num_vec, MPI_DOUBLE, 0,
                       raw_comm );
            Kokkos::deep_copy( x_poly, x_poly_host );
#endif
            auto y_view = Y.getLocalViewDevice();
            Kokkos::parallel_for(
                DTK_MARK_REGION( "polynomial_matrix::apply::no_trans" ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, local_length ),
                KOKKOS_LAMBDA( int const i ) {
                    for ( int j = 0; j < num_vec; ++j )
                        for ( int p = 0; p < poly_size; ++p )
                            y_view( i, j ) +=
                                alpha * vandermonde( i, p ) * x_poly( p, j );
                } );
        }
        else if ( mode == Teuchos::TRANS )
        {
            MultiVector work( Y.getMap(), Y.getNumVectors() );

            // Export X to the polynomial decomposition.
            Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node> exporter(
                X.getMap(), work.getMap() );
            work.doExport( X, exporter, Tpetra::INSERT );

            // Do the local mat-vec.
            auto work_view = work.getLocalViewDevice();
            Kokkos::View<double **, DeviceType> products( "products", poly_size,
                                                          num_vec );
            Kokkos::parallel_for(
                DTK_MARK_REGION( "polynomial_matrix::apply::trans" ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, local_length ),
                KOKKOS_LAMBDA( int const i ) {
                    for ( int j = 0; j < num_vec; ++j )
                        for ( int p = 0; p < poly_size; ++p )
                            products( p, j ) +=
                                alpha * vandermonde( i, p ) * work_view( i, j );
                } );

            // Reduce the results back to the root rank.
            Kokkos::View<double **, DeviceType> product_sums(
                "product_sums", poly_size, num_vec );
#ifdef HAVE_MPI
            auto products_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, products );
            auto product_sums_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, product_sums );
            MPI_Reduce( products_host.data(), product_sums_host.data(),
                        poly_size * num_vec, MPI_DOUBLE, MPI_SUM, 0, raw_comm );
            Kokkos::deep_copy( product_sums, product_sums_host );
#else
            product_sums = products;
#endif

            // Assign the values to Y on the root rank.
            // Note: no alpha here as we used it above.
            if ( 0 == comm->getRank() )
            {
                auto y_view = Y.getLocalViewDevice();

                auto const n = y_view.extent( 0 );
                Kokkos::deep_copy(
                    Kokkos::subview( y_view,
                                     Kokkos::make_pair( n - poly_size, n ),
                                     Kokkos::ALL ),
                    product_sums );
            }
        }
    }

    bool hasTransposeApply() const override { return true; }

  private:
    Kokkos::View<double **, DeviceType> _vandermonde;

    Teuchos::RCP<const Map> _domain_map;
    Teuchos::RCP<const Map> _range_map;
};

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_POLYNOMIALMATRIX_HPP

//---------------------------------------------------------------------------//
// end DTK_PolynomialMatrix.hpp
//---------------------------------------------------------------------------//
