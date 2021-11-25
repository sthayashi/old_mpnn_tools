using LinearAlgebra
using PyCall

using HDF5

function col2row_major(dset)
    return permutedims(dset, reverse(1:ndims(dset)))
end

function get_o2_pos(molecule_group::HDF5.Group, inds)
    o2_pos = read(molecule_group, "o2_pos") |> col2row_major
    o2_pos = o2_pos[inds, :, :]
    return o2_pos
end
    

function make_node_features(atomic_numbers::Vector{Int64})
    """
    Create node features

    Input
    atomic_numbers: Periodic table

    Return
    x: The node features used by pytorch geometric
    """
    # Atoms belonging to organic molecules
    molecule_memberships = ones(eltype(atomic_numbers), length(atomic_numbers))

    # Atoms belonging to O2
    append!(molecule_memberships, [2, 2])

    append!(atomic_numbers, [8, 8])

    # Uncomment the desired node features

    return hcat(molecule_memberships, atomic_numbers)
end

function make_edge_indices(organic_coords, o2_coords; include_intramolecular=false)
    """
    Create the edge indices and features of a directional graph
    between the organic and O2 molecules.
    """
    num_organic_atoms = size(organic_coords, 1)
    organic_indices = [i for i in 1:num_organic_atoms]
    o2_indices = [num_organic_atoms+1, num_organic_atoms+2]

    edge_indices_1 = Vector{Int}()
    edge_indices_2 = Vector{Int}()

    if include_intramolecular == true
        @warn "INTRAMOLECULE EDGES NOT IMPLEMENTED IN JULIA"
#       indices = concatenate(organic_indices, o2_indices)
#       for i,j in permutations(indices, 2)
#           edge_indices_1.append(i)
#           edge_indices_2.append(j)

#       edge_indices_1 = np.array(edge_indices_1)
#       edge_indices_2 = np.array(edge_indices_2)
    else
        for ind_pair in Base.Iterators.product(organic_indices, o2_indices)
            i,j = ind_pair
            append!(edge_indices_1, i)
            append!(edge_indices_2, j)
        end
    end

    return (edge_indices_1, edge_indices_2)
end

function calculate_Cij(i, Z_i, R_i, j, Z_j, R_j)
    if i == j
        @warn "FOUND i==j COULOMB INTERACTION!!"
        return 0.5 * Z_i^2.4
    else
        return (Z_i*Z_j)/norm(R_i-R_j)
    end
end

function make_edge_features(edge_indices_1, edge_indices_2, atomic_numbers, dimer_coords)
    """
    Inputs:
    edge_indices_*: 1D arrays which refer to edges in the graph. E.g.
                    [1, 2], [3, 4] means edges are between (1,3) and (2, 4).

    atomic_numbers: 1D array.
                    The atomic numbers of organic+O2 molecules concatenated.
                    Ordered with dimer_coords.

    dimer_coords:   (number_of_atomsx3) array.
                    All coordinates of all molecules. Order of atoms assumed to match edge_indices.
                    dimer_coords should be the concatenation of organic+O2 molecules.
                    For example, with H2O2, we could have edge indices [1, 2] for H2,
                    and edge indices [3, 4] for O2. We calculate the coulomb interaction
                    for atoms [1, 3] and [2, 4].

    Returns:
    coulomb_interactions: Coulomb interactions as a 1D array, in order of edge_indices_*
    """

    dimer_atomic_numbers = vcat(atomic_numbers, 8, 8)

    ci = Vector{Float32}()
    chi_diff = Vector{Float32}()
    for edge_pair in zip(edge_indices_1, edge_indices_2)
        i, j = edge_pair

        Z_i = dimer_atomic_numbers[i]
        Z_j = dimer_atomic_numbers[j]

        R_i = dimer_coords[i, :]
        R_j = dimer_coords[j, :]
        append!(ci, calculate_Cij(i, Z_i, R_i, j, Z_j, R_j))
        append!(chi_diff, electronegativity_map[atomic_numbers[i]] - 3.44) #3.44 from oxygen
    end

    edge_attr = hcat(ci, chi_diff)
    @assert size(edge_attr) == tuple(length(edge_indices_1), 2) "Edge attributes unexpected shape"
    return edge_attr
end

function fill_edge_attr!(edge_attr, organic_pos, o2_pos, atomic_numbers, edge_indices_i, edge_indices_j)
   #global dimer_atomic_numbers = vcat(atomic_numbers, 8, 8)
    local num_edges = length(edge_indices_i)
    local num_edge_features = 2
   #Threads.@threads for i in eachindex(edge_attr[:, 1, 1])
    Threads.@threads for i in 1:size(edge_attr, 1)
        dimer_atomic_numbers = vcat(atomic_numbers, 8, 8)
        dimer_coords = vcat(organic_pos, o2_pos[i, :, :])
        edge_features = make_edge_features(edge_indices_i, edge_indices_j, dimer_atomic_numbers, dimer_coords)
        edge_features = reshape(edge_features, (num_edges, num_edge_features))
#       @assert size(edge_features) == tuple(length(edge_indices_i), 2) "Edge attributes unexpected shape"
        edge_attr[i, :, :] = edge_features
    end
end

############################################################

mendeleev = pyimport("mendeleev")
global electronegativity_map = Vector{Float32}(undef, 118)
for i in eachindex(electronegativity_map)
    chi = mendeleev.element(i).electronegativity()
    if isnothing(chi)
        electronegativity_map[i] = 0.0
    else
        electronegativity_map[i] = mendeleev.element(Int(i)).electronegativity() 
    end
end

h5py = pyimport("h5py")

fid = h5open("parsed_molecules.h5", "r+")

refcodes = keys(fid)

for refcode in refcodes
    molecule = fid[refcode]

    o2_good_inds = read(molecule, "o2_good_inds") .+ 1 #index shift
    atom_types = read(molecule, "atom_types")
    atomic_numbers = read(molecule, "atomic_numbers")
    organic_pos = read(molecule, "molecule_pos") |> col2row_major

    o2_pos = get_o2_pos(molecule, o2_good_inds)

    node_features = make_node_features(atomic_numbers)
    edge_indices_i, edge_indices_j = make_edge_indices(organic_pos, o2_pos)

    # edge features
    num_edges = length(edge_indices_i)
    num_edge_features = 2
    edge_attr = zeros(Float32, (size(o2_pos, 1), num_edges, num_edge_features))

  # dimer_atomic_numbers = vcat(atomic_numbers, 8, 8)
    # Loop over o2_organic dimer pairs
    fill_edge_attr!(edge_attr, organic_pos, o2_pos, atomic_numbers, edge_indices_i, edge_indices_j)

    molecule["x"] = permutedims(node_features, [2,1])

    molecule["edge_attr"] = permutedims(edge_attr, [3,2,1])
   #molecule["edge_attr"] = edge_attr

    # edge index needs to be shifted back by -1 due to julia -> python
    molecule["edge_indices_i"] = edge_indices_i .- 1
    molecule["edge_indices_j"] = edge_indices_j .- 1
end
