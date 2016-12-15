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
 * \file DTK_CoarseLocalSearch.hpp
 * \author Stuart R. Slattery
 * \brief CoarseLocalSearch declaration.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_COARSELOCALSEARCH_HPP
#define DTK_COARSELOCALSEARCH_HPP

#include <unordered_map>

#include "DTK_EntityIterator.hpp"
#include "DTK_EntityLocalMap.hpp"
#include "DTK_StaticSearchTree.hpp"
#include "DTK_Types.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \class CoarseLocalSearch
 * \brief A CoarseLocalSearch data structure for local entity coarse search.
 */
//---------------------------------------------------------------------------//
class CoarseLocalSearch
{
  public:
    // Constructor.
    CoarseLocalSearch( const EntityIterator &entity_iterator,
                       const Teuchos::RCP<EntityLocalMap> &local_map,
                       const Teuchos::ParameterList &parameters );

    // Find the set of entities a point neighbors.
    void search( const Teuchos::ArrayView<const double> &point,
                 const Teuchos::ParameterList &parameters,
                 Teuchos::Array<Entity> &neighbors ) const;

  private:
    // Local mesh entity centroids.
    Teuchos::Array<double> d_entity_centroids;

    // Local-id to entity map.
    std::unordered_map<int, Entity> d_entity_map;

    // Static search tree.
    Teuchos::RCP<StaticSearchTree> d_tree;
};

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // DTK_COARSELOCALSEARCH_HPP

//---------------------------------------------------------------------------//
// end CoarseLocalSearch.hpp
//---------------------------------------------------------------------------//