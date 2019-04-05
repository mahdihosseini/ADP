import pandas as pd
import numpy as np

def get_htts(level, dataset_type):
    # Load htts at each level
    if dataset_type == 'ADP-Release1':
        htts1 = ['E', 'C', 'H', 'S', 'A', 'M', 'N', 'G', 'T']
        htts2 = ['E.M', 'E.T', 'E.P', 'C.D', 'C.L', 'H.E', 'H.K', 'H.Y', 'S.M', 'S.E', 'S.C', 'S.R', 'A.W',
                 'A.B', 'A.M', 'M.M', 'M.K', 'N.P', 'N.R', 'N.G', 'G.O', 'G.N', 'T']
        htts2p = ['E', 'E.M', 'E.T', 'E.P', 'C', 'C.D', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.E',
                  'S.C', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R', 'N.G', 'G', 'G.O',
                  'G.N', 'T']
        htts3 = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L', 'H.E',
                 'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                 'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W', 'G.O', 'G.N', 'T']
        htts3p = ['E', 'E.M', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C', 'C.D',
                  'C.D.I', 'C.D.R', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.M.C', 'S.M.S', 'S.E', 'S.C',
                  'S.C.H', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K', 'N', 'N.P', 'N.R',
                  'N.R.B', 'N.R.A', 'N.G', 'N.G.M', 'N.G.W', 'G', 'G.O', 'G.N', 'T']
    elif dataset_type == 'ADP-Release1-Flat':
        htts1 = ['E', 'C', 'H', 'S', 'A', 'M', 'N', 'G', 'T']
        htts2 = ['E.M', 'E.T', 'C.D', 'C.L', 'H.E', 'H.K', 'H.Y', 'S.M', 'S.C', 'S.R', 'A.W', 'A.M', 'M', 'N.P',
                 'N.R', 'N.G', 'G.O', 'G.N', 'T']
        htts2p = ['E', 'E.M', 'E.T', 'C', 'C.D', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.C', 'S.R', 'A',
                  'A.W', 'A.M', 'M', 'N', 'N.P', 'N.R', 'N.G', 'G', 'G.O', 'G.N', 'T']
        htts3 = ['E.M.S', 'E.M.C', 'E.T.S', 'E.T.C', 'C.D.I', 'C.D.R', 'C.L', 'H.E', 'H.K', 'H.Y', 'S.M', 'S.C',
                 'S.R', 'A.W', 'A.M', 'M', 'N.P', 'N.R', 'N.G.M', 'G.O', 'G.N', 'T']
        htts3p = ['E', 'E.M', 'E.M.S', 'E.M.C', 'E.T', 'E.T.S', 'E.T.C', 'C', 'C.D', 'C.D.I', 'C.D.R', 'C.L', 'H',
                  'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.C', 'S.R', 'A', 'A.W', 'A.M', 'M', 'N', 'N.P', 'N.R', 'N.G',
                  'N.G.M', 'G', 'G.O', 'G.N', 'T']
    # Validation checks to ensure hand-entered HTTs are valid
    # - That each augmented level is equal to the union of its ancester levels
    assert (sorted([x for x in set(htts1 + htts2)]) == sorted(htts2p)), 'L2+ is not equal to union of L1 and L2'
    assert (sorted([x for x in set(htts1 + htts2 + htts3)]) == sorted(htts3p)), 'L3+ is not equal to union of L1, L2, and L3'

    # Return
    if level == 'L1':
        return htts1, len(htts1)
    elif level == 'L2':
        return htts2, len(htts2)
    elif level == 'L2+':
        return htts2p, len(htts2p)
    elif level == 'L3':
        return htts3, len(htts3)
    elif level == 'L3+':
        return htts3p, len(htts3p)
    return htts, len(htts)