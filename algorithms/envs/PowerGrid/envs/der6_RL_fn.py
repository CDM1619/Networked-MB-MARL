"""""*******************************************************
 * Copyright (C) 2020 {Dong Chen} <{chendon9@msu.edu}>
 * this function is used to interact with RL agent.
"""

from algorithms.envs.PowerGrid.configs.parameters_der6 import *
import numba as nb


@nb.jit
def der_fn(x, t, disturbance_R, disturbance_L):
    # ----------Transferring Inv. Output Currents to Global DQ-----------------
    # dq --> \alpha \beta
    ioD1 = math.cos(0) * x[4] - math.sin(0) * x[5]
    ioQ1 = math.sin(0) * x[4] + math.cos(0) * x[5]
    ioD2 = math.cos(x[6]) * x[9] - math.sin(x[6]) * x[10]
    ioQ2 = math.sin(x[6]) * x[9] + math.cos(x[6]) * x[10]
    ioD3 = math.cos(x[11]) * x[14] - math.sin(x[11]) * x[15]
    ioQ3 = math.sin(x[11]) * x[14] + math.cos(x[11]) * x[15]
    ioD4 = math.cos(x[16]) * x[19] - math.sin(x[16]) * x[20]
    ioQ4 = math.sin(x[16]) * x[19] + math.cos(x[16]) * x[20]
    ioD5 = math.cos(x[21]) * x[24] - math.sin(x[21]) * x[25]
    ioQ5 = math.sin(x[21]) * x[24] + math.cos(x[21]) * x[25]
    ioD6 = math.cos(x[26]) * x[29] - math.sin(x[26]) * x[30]
    ioQ6 = math.sin(x[26]) * x[29] + math.cos(x[26]) * x[30]

    # ----------Defining Bus Voltages-----------------
    vbD1 = rN * (ioD1 - x[31])
    vbQ1 = rN * (ioQ1 - x[32])
    vbD2 = rN * (x[31] + x[49] - x[33])
    vbQ2 = rN * (x[32] + x[50] - x[34])
    vbD3 = rN * (ioD2 - x[49])
    vbQ3 = rN * (ioQ2 - x[50])
    vbD4 = rN * (x[33] + x[35] - x[37] - x[51])
    vbQ4 = rN * (x[34] + x[36] - x[38] - x[52])
    vbD5 = rN * (ioD4 - x[35])
    vbQ5 = rN * (ioQ4 - x[36])
    vbD6 = rN * (x[37] + x[39] - x[41])
    vbQ6 = rN * (x[38] + x[40] - x[42])
    vbD7 = rN * (ioD6 + x[41] - x[53] - x[43])
    vbQ7 = rN * (ioQ6 + x[42] - x[54] - x[44])
    vbD9 = rN * (x[43] + x[45] - x[55])
    vbQ9 = rN * (x[44] + x[46] - x[56])
    vbD11 = rN * (x[47] - x[45] - x[57])
    vbQ11 = rN * (x[48] - x[46] - x[58])
    vbD13 = rN * (ioD5 - x[47])
    vbQ13 = rN * (ioQ5 - x[48])
    vbD14 = rN * (ioD3 - x[39])
    vbQ14 = rN * (ioQ3 - x[40])

    # ----------Transferring Bus Voltages to Inv. dq-----------------
    # \alpha \beta --> dq
    vbd1 = math.cos(0) * vbD1 + math.sin(0) * vbQ1
    vbq1 = -math.sin(0) * vbD1 + math.cos(0) * vbQ1
    vbd3 = math.cos(x[6]) * vbD3 + math.sin(x[6]) * vbQ3
    vbq3 = -math.sin(x[6]) * vbD3 + math.cos(x[6]) * vbQ3
    vbd14 = math.cos(x[11]) * vbD14 + math.sin(x[11]) * vbQ14
    vbq14 = -math.sin(x[11]) * vbD14 + math.cos(x[11]) * vbQ14
    vbd5 = math.cos(x[16]) * vbD5 + math.sin(x[16]) * vbQ5
    vbq5 = -math.sin(x[16]) * vbD5 + math.cos(x[16]) * vbQ5
    vbd13 = math.cos(x[21]) * vbD13 + math.sin(x[21]) * vbQ13
    vbq13 = -math.sin(x[21]) * vbD13 + math.cos(x[21]) * vbQ13
    vbd7 = math.cos(x[26]) * vbD7 + math.sin(x[26]) * vbQ7
    vbq7 = -math.sin(x[26]) * vbD7 + math.cos(x[26]) * vbQ7

    wcom = x[59] - mp1 * x[2]
    ## ----------------DG1--------------------
    vod1_star = x[65] - nq1 * x[3]
    voq1_star = 0
    xdot1 = x[59] - mp1 * x[2] - wcom
    xdot2 = wc * (vod1_star * x[4] + voq1_star * x[5] - x[2])
    xdot3 = wc * (-vod1_star * x[5] + voq1_star * x[4] - x[3])
    xdot4 = (-rLc / Lc) * x[4] + wcom * x[5] + (1 / Lc) * (vod1_star - vbd1)
    xdot5 = (-rLc / Lc) * x[5] - wcom * x[4] + (1 / Lc) * (voq1_star - vbq1)

    ## ----------------DG2-------------------
    vod2_star = x[66] - nq2 * x[8]
    voq2_star = 0
    xdot6 = x[60] - mp2 * x[7] - wcom
    xdot7 = wc * (vod2_star * x[9] + voq2_star * x[10] - x[7])
    xdot8 = wc * (-vod2_star * x[10] + voq2_star * x[9] - x[8])
    xdot9 = (-rLc / Lc) * x[9] + wcom * x[10] + (1 / Lc) * (vod2_star - vbd3)
    xdot10 = (-rLc / Lc) * x[10] - wcom * x[9] + (1 / Lc) * (voq2_star - vbq3)

    ## ----------------DG3-------------------
    vod3_star = x[67] - nq3 * x[13]
    voq3_star = 0
    xdot11 = x[61] - mp3 * x[12] - wcom
    xdot12 = wc * (vod3_star * x[14] + voq3_star * x[15] - x[12])
    xdot13 = wc * (-vod3_star * x[15] + voq3_star * x[14] - x[13])
    xdot14 = (-rLc / Lc) * x[14] + wcom * x[15] + (1 / Lc) * (vod3_star - vbd14)
    xdot15 = (-rLc / Lc) * x[15] - wcom * x[14] + (1 / Lc) * (voq3_star - vbq14)

    ## ----------------DG4-------------------
    vod4_star = x[68] - nq4 * x[18]
    voq4_star = 0
    xdot16 = x[62] - mp4 * x[17] - wcom
    xdot17 = wc * (vod4_star * x[19] + voq4_star * x[20] - x[17])
    xdot18 = wc * (-vod4_star * x[20] + voq4_star * x[19] - x[18])
    xdot19 = (-rLc / Lc) * x[19] + wcom * x[20] + (1 / Lc) * (vod4_star - vbd5)
    xdot20 = (-rLc / Lc) * x[20] - wcom * x[19] + (1 / Lc) * (voq4_star - vbq5)

    # ----------------DG5-------------------
    vod5_star = x[69] - nq5 * x[23]
    voq5_star = 0
    xdot21 = x[63] - mp5 * x[22] - wcom
    xdot22 = wc * (vod5_star * x[24] + voq5_star * x[25] - x[22])
    xdot23 = wc * (-vod5_star * x[25] + voq5_star * x[24] - x[23])
    xdot24 = (-rLc / Lc) * x[24] + wcom * x[25] + (1 / Lc) * (vod5_star - vbd13)
    xdot25 = (-rLc / Lc) * x[25] - wcom * x[24] + (1 / Lc) * (voq5_star - vbq13)

    # ----------------DG6-------------------
    vod6_star = x[70] - nq6 * x[28]
    voq6_star = 0
    xdot26 = x[64] - mp6 * x[27] - wcom
    xdot27 = wc * (vod6_star * x[29] + voq6_star * x[30] - x[27])
    xdot28 = wc * (-vod6_star * x[30] + voq6_star * x[29] - x[28])
    xdot29 = (-rLc / Lc) * x[29] + wcom * x[30] + (1 / Lc) * (vod6_star - vbd7)
    xdot30 = (-rLc / Lc) * x[30] - wcom * x[29] + (1 / Lc) * (voq6_star - vbq7)

    # ----------------------------------------------------
    # ---------line1
    xdot31 = (-rline1 / Lline1) * x[31] + wcom * x[32] + (1 / Lline1) * (vbD1 - vbD2)
    xdot32 = (-rline1 / Lline1) * x[32] - wcom * x[31] + (1 / Lline1) * (vbQ1 - vbQ2)
    # ---------line2
    xdot33 = (-rline2 / Lline2) * x[33] + wcom * x[34] + (1 / Lline2) * (vbD2 - vbD4)
    xdot34 = (-rline2 / Lline2) * x[34] - wcom * x[33] + (1 / Lline2) * (vbQ2 - vbQ4)
    # ---------line3
    xdot35 = (-rline3 / Lline3) * x[35] + wcom * x[36] + (1 / Lline3) * (vbD5 - vbD4)
    xdot36 = (-rline3 / Lline3) * x[36] - wcom * x[35] + (1 / Lline3) * (vbQ5 - vbQ4)
    # ---------line4
    xdot37 = (-rline4 / Lline4) * x[37] + wcom * x[38] + (1 / Lline4) * (vbD4 - vbD6)
    xdot38 = (-rline4 / Lline4) * x[38] - wcom * x[37] + (1 / Lline4) * (vbQ4 - vbQ6)
    # ---------line5
    xdot39 = (-rline5 / Lline5) * x[39] + wcom * x[40] + (1 / Lline5) * (vbD14 - vbD6)
    xdot40 = (-rline5 / Lline5) * x[40] - wcom * x[39] + (1 / Lline5) * (vbQ14 - vbQ6)
    # ---------line6
    xdot41 = (-rline6 / Lline6) * x[41] + wcom * x[42] + (1 / Lline6) * (vbD6 - vbD7)
    xdot42 = (-rline6 / Lline6) * x[42] - wcom * x[41] + (1 / Lline6) * (vbQ6 - vbQ7)
    # ---------line8
    xdot43 = (-rline8 / Lline8) * x[43] + wcom * x[44] + (1 / Lline8) * (vbD7 - vbD9)
    xdot44 = (-rline8 / Lline8) * x[44] - wcom * x[43] + (1 / Lline8) * (vbQ7 - vbQ9)
    # ---------line10
    xdot45 = (-rline10 / Lline10) * x[45] + wcom * x[46] + (1 / Lline10) * (vbD11 - vbD9)
    xdot46 = (-rline10 / Lline10) * x[46] - wcom * x[45] + (1 / Lline10) * (vbQ11 - vbQ9)
    # ---------line12
    xdot47 = (-rline12 / Lline12) * x[47] + wcom * x[48] + (1 / Lline12) * (vbD13 - vbD11)
    xdot48 = (-rline12 / Lline12) * x[48] - wcom * x[47] + (1 / Lline12) * (vbQ13 - vbQ11)
    # ---------line13
    xdot49 = (-rline13 / Lline13) * x[49] + wcom * x[50] + (1 / Lline13) * (vbD3 - vbD2)
    xdot50 = (-rline13 / Lline13) * x[50] - wcom * x[49] + (1 / Lline13) * (vbQ3 - vbQ2)

    # -------------------------Loads------------------
    # ------Load1--------
    xdot51 = (-(disturbance_R[0] * Rload1) / (disturbance_L[0] * Lload1)) * x[51] + wcom * x[52] + (
            1 / (disturbance_L[0] * Lload1)) * vbD4
    xdot52 = (-(disturbance_R[0] * Rload1) / (disturbance_L[0] * Lload1)) * x[52] - wcom * x[51] + (
            1 / (disturbance_L[0] * Lload1)) * vbQ4
    # ------Load2--------
    xdot53 = (-(disturbance_R[1] * Rload2 + rline7) / (disturbance_L[1] * Lload2 + Lline7)) * \
             x[53] + wcom * x[54] + (1 / (disturbance_L[1] * Lload2 + Lline7)) * vbD7
    xdot54 = (-(disturbance_R[1] * Rload2 + rline7) / (disturbance_L[1] * Lload2 + Lline7)) * \
             x[54] - wcom * x[53] + (1 / (disturbance_L[1] * Lload2 + Lline7)) * vbQ7
    # ------Load3--------
    xdot55 = (-(disturbance_R[2] * Rload3 + rline9) / (disturbance_L[2] * Lload3 + Lline9)) * \
             x[55] + wcom * x[56] + (1 / (disturbance_L[2] * Lload3 + Lline9)) * vbD9
    xdot56 = (-(disturbance_R[2] * Rload3 + rline9) / (disturbance_L[2] * Lload3 + Lline9)) * \
             x[56] - wcom * x[55] + (1 / (disturbance_L[2] * Lload3 + Lline9)) * vbQ9
    # ------Load4--------
    xdot57 = (-(disturbance_R[3] * Rload4 + rline11) / (disturbance_L[3] * Lload4 + Lline11)) * \
             x[57] + wcom * x[58] + (1 / (disturbance_L[3] * Lload4 + Lline11)) * vbD11
    xdot58 = (-(disturbance_R[3] * Rload4 + rline11) / (disturbance_L[3] * Lload4 + Lline11)) * \
             x[58] - wcom * x[57] + (1 / (disturbance_L[3] * Lload4 + Lline11)) * vbQ11

    if t <= 0.4:
        # this time is for the primary control
        xdot59 = 0
        xdot60 = 0
        xdot61 = 0
        xdot62 = 0
        xdot63 = 0
        xdot64 = 0
        xdot65 = 0
        xdot66 = 0
        xdot67 = 0
        xdot68 = 0
        xdot69 = 0
        xdot70 = 0
    else:
        # --------Controller Parameters--------
        w_array = np.array([[x[59] - mp1 * x[2]],
                            [x[60] - mp2 * x[7]],
                            [x[61] - mp3 * x[12]],
                            [x[62] - mp4 * x[17]],
                            [x[63] - mp5 * x[22]],
                            [x[64] - mp6 * x[27]]])

        Pratio = np.array([[mp1 * x[2]], [mp2 * x[7]], [mp3 * x[12]], [mp4 * x[17]], [mp5 * x[22]], [mp6 * x[27]]])

        # Conventional Freq Control
        Synch_Mat = -1 * a_ctrl * (np.dot(L + G, w_array - np.array([[wref], [wref], [wref], [wref], [wref], [wref]]))
                                   + np.dot(L, Pratio))

        xdot59 = Synch_Mat[0][0]
        xdot60 = Synch_Mat[1][0]
        xdot61 = Synch_Mat[2][0]
        xdot62 = Synch_Mat[3][0]
        xdot63 = Synch_Mat[4][0]
        xdot64 = Synch_Mat[5][0]
        xdot65 = 0
        xdot66 = 0
        xdot67 = 0
        xdot68 = 0
        xdot69 = 0
        xdot70 = 0

    return np.array(
        [0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6, xdot7, xdot8, xdot9, xdot10, xdot11, xdot12, xdot13, xdot14,
         xdot15, xdot16, xdot17, xdot18, xdot19, xdot20, xdot21, xdot22, xdot23, xdot24, xdot25, xdot26, xdot27,
         xdot28, xdot29, xdot30, xdot31, xdot32, xdot33, xdot34, xdot35, xdot36, xdot37, xdot38, xdot39, xdot40,
         xdot41, xdot42, xdot43, xdot44, xdot45, xdot46, xdot47, xdot48, xdot49, xdot50, xdot51, xdot52, xdot53,
         xdot54, xdot55, xdot56, xdot57, xdot58, xdot59, xdot60, xdot61, xdot62, xdot63, xdot64, xdot65, xdot66,
         xdot67, xdot68, xdot69, xdot70, ioD1, ioQ1, vbD1, vbQ1, ioD2, ioQ2, vbD3, vbQ3, ioD3, ioQ3, vbD14, vbQ14, ioD4,
         ioQ4, vbD5, vbQ5, ioD5, ioQ5, vbD13, vbQ13, ioD6, ioQ6, vbD7, vbQ7])
