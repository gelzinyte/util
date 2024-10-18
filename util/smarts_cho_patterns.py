smartsPatt = {
    1: ('[O;!H0]', 0),  # OH
    2: ('[!#6;!#1;!H0]', 1),  # QH > 1
    3: ('[OD2;!$(OC=O)]', 0), # ether, not esther
    4: ("[#8]=[CH]", 0), # aldehyde?
    5:('[#6]-[#6](=[#8])-[#6]', 0), # ketone
    6: ('[#6]-[#6](=[#8])-[#8]-[#1]', 0), #carboxylic acid
    7: ('[#6]-[#6](=[#8])-[#8]-[#6]', 0), # esther
    8: ('c-O', 0), # phenol
    9: ('[#8]~[#6]~[#8]', 0),  # OCO
    10: ('[#8]~*~*~[#8]', 0),  # OAAO   
    11: ('[!#6;!#1]~[CH3]', 0),  # QCH3 # OCH3
    # carbon 
    12:('[#6]=[#6]', 0) , # C=C
    13: ('[CH2]=*', 0),  # CH2=A
    # 13: ('[#6]=[#6](~[#6])~[#6]', 0),  # C=C(C)C
    14: ('[#6]#[#6]', 0),  #CTC
    15: ('[CH]#[#6]', 0),  #Terminal tripple bond
    16:('[C;H3,H4]', 0),  #CH3
    17: ('[CH3]~[CH2]~*', 0),  # CH3CH2A            # ethyl group - aliphatic CH2 
    18: ('[CH3]~[CH2]~[CH2]*', 0),  # CH3CH2A            # ethyl group - aliphatic CH2 
    19:('a', 0),  # Aromatic
    # rings                  
    20: ('*1~*~*~1', 0),  # 3M Ring
    21: ('*1~*~*~*~1', 0),  # 4M Ring
    22: ('*1~*~*~*~*~1', 0),  # 5 M ring
    23: ('*1~*~*~*~*~*~1', 0),  # 6M Ring
    24: ('*1~*~*~*~*~*~1', 1),  # 6M ring > 1
    25: ('*1~*~*~*~*~*~*~1', 0),  # 7M Ring
    26: ('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]', 0),  # 8M Ring or larger. This only handles up to ring sizes of 14 (napthalene would count)
    27: ('[!C;!c;R]', 0),  # Heterocycle
    28: ('[!#6;!#1]1~*~*~*~*~*~1', 0)  # QAAAAA@1
}

labels = {
    1: '-OH',
    2: 'Multiple -OH',
    3: 'Ether',
    4: 'Aldehyde',
    5: 'Ketone',
    6: 'Carboxylic Acid',
    7: 'Ester',
    8: 'Phenol',
    9: '-O~C~O-',
    10: '-O~C~C~O-',
    11: '-OMe',
    12: '-C=C-',
    13: '-C=CH2',
    14: '-C#C-',
    15: '-C#CH',
    16: 'Methyl',
    17: 'Ethyl',
    18: 'Propyl',
    19: 'Aromaticity',
    20: '3m Ring',
    21: '4m Ring',
    22: '5m Ring',
    23: '6m Ring',
    24: 'Multiple 6m Rings',
    25: '7m Ring',
    26: '8m Ring or larger',
    27: 'Any Heterocycle',
    28: '6m Heterocycle'
}
