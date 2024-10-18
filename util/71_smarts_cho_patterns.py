smartsPatt = {
    # rings                  
    1: ('*1~*~*~1', 0),  # 3M Ring
    2: ('*1~*~*~*~1', 0),  # 4M Ring
    3: ('*1~*~*~*~*~1', 0),  # 5 M ring
    4: ('*1~*~*~*~*~*~1', 0),  # 6M Ring
    5: ('*1~*~*~*~*~*~1', 1),  # 6M ring > 1
    6: ('*1~*~*~*~*~*~*~1', 0),  # 7M Ring
    7: ('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]', 0),  # 8M Ring or larger. This only handles up to ring sizes of 14 (napthalene would count)
    # misc
    8:('a', 0),  # Aromatic
    # oxygen based
    9: ('[O;!H0]', 0),  # OH
    10: ('[#8]~[#6](~[#8])~[#8]', 0),  # OC(O)O
    11: ('[#8]~[#6]~[#8]', 0),  # OCO
    # carbon 
    12: ('[#6]#[#6]', 0),  #CTC
    13: ('[#6]=[#6](~[#6])~[#6]', 0),  # C=C(C)C
    14: ('[CH3]~*~[CH3]', 0),  # CH3ACH3
    15:('[C;H3,H4]', 0),  #CH3
    # my 
    16: ('[OD2;!$(OC=O)]', 0), # ether, not esther
    17: ("[#8]=[CH]", 0), # aldehyde?
    18:('[#6]-[#6](=[#8])-[#6]', 0), # ketone
    19: ('[#6]-[#6](=[#8])-[#8]-[#1]', 0), #carboxylic acid
    20: ('[#6]-[#6](=[#8])-[#8]-[#6]', 0), # esther
    21: ('c-O', 0), # phenol
    # weird/ununderstood/shouldn't be present
    22: ('[!#6;!#1]1~*~*~*~1', 0),  # QAAA@1
    23:('[!#6;!#1]1~*~*~1', 0),  # QAA@1
    24:('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 0),  # C=C(Q)Q
    25:('[#6]=;@[#6](@*)@*', 0),  # C$=C($A)$A
    26: ('[!#6;!#1]~[CH2]~[!#6;!#1]', 0),  # QCH2Q
    27:('[#6]~[!#6;!#1](~[#6])(~[#6])~*', 0),  # CQ(C)(C)A
    28:('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 0),  # QHAQH
    29: ('[CH2]=*', 0),  # CH2=A
    30:('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]', 0),  # QHAAAQH
    31: ('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 0),  # QHAAQH
    32: ('[!#6;!#1]~[!#6;!#1;!H0]', 0),  # QQH
    33: ('[#16]!:*:*', 0),  # Snot%A%A
    34: ('*@*!@*@*', 0),  # A$!A$A
    35: ('[#8]~*~*~[#8]', 0),  # OAAO   
    36: ('*~[CH2]~[!#6;!#1;!H0]', 0),  # ACH2QH
    37: ('[C;H2,H3][!#6;!#1][C;H2,H3]', 0),  # CH2QCH2 
    38: ('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]', 0),  # QHAACH2A
    39: ('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]', 0),  # QHAAACH2A
    40: ('[!#6;!#1]~[CH3]', 0),  # QCH3 # OCH3
    41: ('[!#6;!#1]1~*~*~*~*~*~1', 0),  # QAAAAA@1
    42: ('[!#6;!#1]~[#8]', 0),  # QO
    43: ('[!#6;!#1;!H0]~*~[CH2]~*', 0),  # QHACH2A
    44: ('*@*(@*)@*', 0),  # A$A($A)$A              eg napthalene - two rings meeting? 
    45: ('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 0),  # QA(Q)Q
    46: ('[#8]!:*:*', 0),  # Onot%A%A
    47: ('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]', 1),  # ACH2CH2A > 1
    48: ('[!#6;R]', 1),  # Heterocyclic atom > 1 (&...) Spec Incomplete
    49: ('[!#6;!#1]~[!#6;!#1]', 0),  # QQ
    50: ('*!@[#8]!@*', 0),  # A!O!A     eg COC non ring
    51: ('*@*!@[#8]', 1),
    52: ('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]', 0),  # ACH2AAACH2A
    53: ('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]', 0),  # ACH2AACH2A
    54: ('[!#6;!#1]~[!#6;!#1]', 1),  # QQ > 1 (&...)  Spec Incomplete
    55: ('[!#6;!#1;!H0]', 1),  # QH > 1
    56: ('[#8]~*~[CH2]~*', 0),  # OACH2A            eg matches CC(=O)CC
    57: ('[!C;!c;R]', 0),  # Heterocycle
    58: ('[!#6;!#1]~[CH2]~*', 1),  # QCH2A>1 (&...) Spec Incomplete # O-Ch2-H/C
    59: ('*!:*:*!:*', 0),  # Anot%A%Anot%A
    60: ('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 0),  # ACH2CH2A
    61: ('*!@*@*!@*', 0),  # A!A$A!A
    62: ('[#8]~[#6](~[#6])~[#6]', 0),  # OC(C)C     eg matches CC(=O)CC 
    63: ('*!@[CH2]!@*', 0),  # A!CH2!A              eg matches CC(=O)CC - aliphatic CH2 
    # oxygen-based
    64: ('*~[CH2]~[#8]', 0),  # ACH2O               aliphatic CH2 next to Oxygen
    65:('*@*!@[#8]', 0),  # A$A!O
    # carbon-based
    66: ('[#6]~[#6](~[#6])(~[#6])~*', 0),  # CC(C)(C)A
    67:('[#6]=[#6](~*)~*', 0),  # C=C(A)A
    68: ('[#6]=[#6]', 0),  # C=C (non-aromatic?)
    69: ('[CH3]~*~*~*~[CH2]~*', 0),  # CH3AAACH2A
    70: ('*~*(~*)(~*)~*', 0),  # AA(A)(A)A
    71: ('[CH3]~[CH2]~*', 0),  # CH3CH2A            # ethyl group - aliphatic CH2 
}

labels = {
    1: "3m ring",
    2: "4m ring",
    3: "5m ring",
    4: "6m ring",
    5: "6m ring > 1",
    6: "7m ring",
    7: "7m+ ring",
    8: "aromatic",
    9: "OH",
    10: "OC(O)O",
    11: "OCO",
    12: "C#C",
    13: "C=C(C)C",
    14: "CH3-any-CH3",
    15: "CH3",
    16: "ether",
    17: "aldehyde",
    18: "ketone",
    19: "carboxylic acid",
    20: "esther",
    21: "phenol",
    22: "4m heteroatom ring",
    23: "3m heteroatom ring",
    24: "C=C(Q)Q",
    25: "C$=C($A)$A", #??
    26: "QCH2Q", 
    27: "CQ(C)(C)A",
    28: "QHAQH",
    29: "CH2=A",
    30: "QHAAAQH",
    31: "QHAAQH",
    32: "QQH",
    33: "Snot%A%A",
    34: "A$!A$A",
    35: "OAAO",
    36: "ACH2QH",
    37: "CH2QCH2",
    38: "QHAACH2A",
    39: "QHAAACH2A",
    40: "QCH3",
    41: "6m heteroatom ring",
    42: "QO",
    43: "QHACH2A",
    44: "A$A($A)$A",
    45: "QA(Q)Q",
    46: "Snot%A%A",
    47: "ACH2CH2A > 1",
    48: "heterocyclic atom > 1",
    49: "QQ > 1",
    50: "A!O!A",
    51: "A$A!O",
    52: "ACH2AAACH2A",
    53: "ACH2AACH2A",
    54: "QQ > 1",
    55: "QH > 1",
    56: "OACH2A",
    57: "heterocycle",
    58: "QCH2A>1",
    59: "Anot%A%Anot%A",
    60: "ACH2CH2A",
    61: "A!A$A!A",
    62: "OC(C)C",
    63: "A!CH2!A",
    64: "ACH2O",
    65: "A$A!O",
    66: "CC(C)(C)A",
    67: "C=C(A)A",
    68: "C=C",
    69: "CH3AAACH2A",
    70: "AA(A)(A)A",
    71: "CH3CH2A"
}
