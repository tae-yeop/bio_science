## Alphafold 2
two major components : Evoformer + Structure Module
Evoformer : extracts information from the MSA using attention mechanisms into an “MSA representation” which is used to update a "pair representation".
This pair representation represents the edges of a graph
Each edge contains information about the distance between two residues
frame aligned point error loss : comparing the predicted and ground truth atomic co-ordinates.
distogram loss : ensuring the pair representation does indeed learn structural properties of the protein
MSA loss : masks the input MSA and attempts to predict the masked residues

## Alphafold 3

two major components : Pairformer + Diffusion Module
Pairformer : simpler (and more efficient) version of the Evoformer, which no longer directly processes the MSA representation
Diffusion Module : starts with a point cloud of random noise, refining the co-ordinates based on the information extracted by the Pairformer
The primary loss is the diffusion loss, once again operating on the final atomic co-ordinates.
The distogram loss remains from Alphafold 2