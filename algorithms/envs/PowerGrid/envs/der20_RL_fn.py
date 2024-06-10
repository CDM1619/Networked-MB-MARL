"""""*******************************************************
 * Copyright (C) 2020 {Dong Chen} <{chendon9@msu.edu}>
 * this function is used to interact with RL agent.
"""

from algorithms.envs.PowerGrid.configs.parameters_der20 import *
import numba as nb


@nb.jit
def der_fn(x, t, disturbance_R, disturbance_L):
    # ---------Transferring Inv. Output Currents to Global DQ-----------------
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
    ioD7 = math.cos(x[31]) * x[34] - math.sin(x[31]) * x[35]
    ioQ7 = math.sin(x[31]) * x[34] + math.cos(x[31]) * x[35]
    ioD8 = math.cos(x[36]) * x[39] - math.sin(x[36]) * x[40]
    ioQ8 = math.sin(x[36]) * x[39] + math.cos(x[36]) * x[40]
    ioD9 = math.cos(x[41]) * x[44] - math.sin(x[41]) * x[45]
    ioQ9 = math.sin(x[41]) * x[44] + math.cos(x[41]) * x[45]
    ioD10 = math.cos(x[46]) * x[49] - math.sin(x[46]) * x[50]
    ioQ10 = math.sin(x[46]) * x[49] + math.cos(x[46]) * x[50]
    ioD11 = math.cos(x[51]) * x[54] - math.sin(x[51]) * x[55]
    ioQ11 = math.sin(x[51]) * x[54] + math.cos(x[51]) * x[55]
    ioD12 = math.cos(x[56]) * x[59] - math.sin(x[56]) * x[60]
    ioQ12 = math.sin(x[56]) * x[59] + math.cos(x[56]) * x[60]
    ioD13 = math.cos(x[61]) * x[64] - math.sin(x[61]) * x[65]
    ioQ13 = math.sin(x[61]) * x[64] + math.cos(x[61]) * x[65]
    ioD14 = math.cos(x[66]) * x[69] - math.sin(x[66]) * x[70]
    ioQ14 = math.sin(x[66]) * x[69] + math.cos(x[66]) * x[70]
    ioD15 = math.cos(x[71]) * x[74] - math.sin(x[71]) * x[75]
    ioQ15 = math.sin(x[71]) * x[74] + math.cos(x[71]) * x[75]
    ioD16 = math.cos(x[76]) * x[79] - math.sin(x[76]) * x[80]
    ioQ16 = math.sin(x[76]) * x[79] + math.cos(x[76]) * x[80]
    ioD17 = math.cos(x[81]) * x[84] - math.sin(x[81]) * x[85]
    ioQ17 = math.sin(x[81]) * x[84] + math.cos(x[81]) * x[85]
    ioD18 = math.cos(x[86]) * x[89] - math.sin(x[86]) * x[90]
    ioQ18 = math.sin(x[86]) * x[89] + math.cos(x[86]) * x[90]
    ioD19 = math.cos(x[91]) * x[94] - math.sin(x[91]) * x[95]
    ioQ19 = math.sin(x[91]) * x[94] + math.cos(x[91]) * x[95]
    ioD20 = math.cos(x[96]) * x[99] - math.sin(x[96]) * x[100]
    ioQ20 = math.sin(x[96]) * x[99] + math.cos(x[96]) * x[100]

    # ---------Defining Bus Voltages-----------------
    vbD1 = rN * (ioD1 - x[141] - x[101])
    vbQ1 = rN * (ioQ1 - x[142] - x[102])
    vbD2 = rN * (ioD2 + x[105] - x[107])
    vbQ2 = rN * (ioQ2 + x[106] - x[108])
    vbD3 = rN * (ioD3 + x[107] - x[109] - x[145])
    vbQ3 = rN * (ioQ3 + x[108] - x[110] - x[146])
    vbD4 = rN * (ioD4 + x[113] - x[115])
    vbQ4 = rN * (ioQ4 + x[114] - x[116])
    vbD5 = rN * (ioD5 + x[115] - x[117] - x[147])
    vbQ5 = rN * (ioQ5 + x[116] - x[118] - x[148])
    vbD6 = rN * (ioD6 + x[101] - x[103])
    vbQ6 = rN * (ioQ6 + x[102] - x[104])
    vbD7 = rN * (ioD7 + x[103] - x[105] - x[143])
    vbQ7 = rN * (ioQ7 + x[104] - x[106] - x[144])
    vbD8 = rN * (ioD8 + x[109] - x[111] - x[119])
    vbQ8 = rN * (ioQ8 + x[110] - x[112] - x[120])
    vbD9 = rN * (ioD9 + x[111] - x[113] - x[147])
    vbQ9 = rN * (ioQ9 + x[112] - x[114] - x[148])
    vbD10 = rN * (ioD10 + x[117] - x[119])
    vbQ10 = rN * (ioQ10 + x[118] - x[120])
    vbD11 = rN * (ioD11 + x[119] - x[123] - x[151])
    vbQ11 = rN * (ioQ11 + x[120] - x[124] - x[152])
    vbD12 = rN * (ioD12 + x[127] - x[129])
    vbQ12 = rN * (ioQ12 + x[128] - x[130])
    vbD13 = rN * (ioD13 + x[121] + x[129] - x[131] - x[155])
    vbQ13 = rN * (ioQ13 + x[122] + x[130] - x[132] - x[156])
    vbD14 = rN * (ioD14 + x[135] - x[137])
    vbQ14 = rN * (ioQ14 + x[136] - x[138])
    vbD15 = rN * (ioD15 + x[137] - x[139] - x[159])
    vbQ15 = rN * (ioQ15 + x[138] - x[140] - x[160])
    vbD16 = rN * (ioD16 + x[123] - x[125])
    vbQ16 = rN * (ioQ16 + x[124] - x[126])
    vbD17 = rN * (ioD17 + x[125] - x[127] - x[153])
    vbQ17 = rN * (ioQ17 + x[126] - x[128] - x[154])
    vbD18 = rN * (ioD18 + x[131] - x[133])
    vbQ18 = rN * (ioQ18 + x[132] - x[134])
    vbD19 = rN * (ioD19 + x[133] - x[135] - x[157])
    vbQ19 = rN * (ioQ19 + x[134] - x[136] - x[158])
    vbD20 = rN * (ioD20 + x[139])
    vbQ20 = rN * (ioQ20 + x[140])

    # ---------Transferring Bus Voltages to Inv. dq-----------------
    vbd1 = math.cos(0) * vbD1 + math.sin(0) * vbQ1
    vbq1 = -math.sin(0) * vbD1 + math.cos(0) * vbQ1
    vbd2 = math.cos(x[6]) * vbD2 + math.sin(x[6]) * vbQ2
    vbq2 = -math.sin(x[6]) * vbD2 + math.cos(x[6]) * vbQ2
    vbd3 = math.cos(x[11]) * vbD3 + math.sin(x[11]) * vbQ3
    vbq3 = -math.sin(x[11]) * vbD3 + math.cos(x[11]) * vbQ3
    vbd4 = math.cos(x[16]) * vbD4 + math.sin(x[16]) * vbQ4
    vbq4 = -math.sin(x[16]) * vbD4 + math.cos(x[16]) * vbQ4
    vbd5 = math.cos(x[21]) * vbD5 + math.sin(x[21]) * vbQ5
    vbq5 = -math.sin(x[21]) * vbD5 + math.cos(x[21]) * vbQ5
    vbd6 = math.cos(x[26]) * vbD6 + math.sin(x[26]) * vbQ6
    vbq6 = -math.sin(x[26]) * vbD6 + math.cos(x[26]) * vbQ6
    vbd7 = math.cos(x[31]) * vbD7 + math.sin(x[31]) * vbQ7
    vbq7 = -math.sin(x[31]) * vbD7 + math.cos(x[31]) * vbQ7
    vbd8 = math.cos(x[36]) * vbD8 + math.sin(x[36]) * vbQ8
    vbq8 = -math.sin(x[36]) * vbD8 + math.cos(x[36]) * vbQ8
    vbd9 = math.cos(x[41]) * vbD9 + math.sin(x[41]) * vbQ9
    vbq9 = -math.sin(x[41]) * vbD9 + math.cos(x[41]) * vbQ9
    vbd10 = math.cos(x[46]) * vbD10 + math.sin(x[46]) * vbQ10
    vbq10 = -math.sin(x[46]) * vbD10 + math.cos(x[46]) * vbQ10
    vbd11 = math.cos(x[51]) * vbD11 + math.sin(x[51]) * vbQ11
    vbq11 = -math.sin(x[51]) * vbD11 + math.cos(x[51]) * vbQ11
    vbd12 = math.cos(x[56]) * vbD12 + math.sin(x[56]) * vbQ12
    vbq12 = -math.sin(x[56]) * vbD12 + math.cos(x[56]) * vbQ12
    vbd13 = math.cos(x[61]) * vbD13 + math.sin(x[61]) * vbQ13
    vbq13 = -math.sin(x[61]) * vbD13 + math.cos(x[61]) * vbQ13
    vbd14 = math.cos(x[66]) * vbD14 + math.sin(x[66]) * vbQ14
    vbq14 = -math.sin(x[66]) * vbD14 + math.cos(x[66]) * vbQ14
    vbd15 = math.cos(x[71]) * vbD15 + math.sin(x[71]) * vbQ15
    vbq15 = -math.sin(x[71]) * vbD15 + math.cos(x[71]) * vbQ15
    vbd16 = math.cos(x[76]) * vbD16 + math.sin(x[76]) * vbQ16
    vbq16 = -math.sin(x[76]) * vbD16 + math.cos(x[76]) * vbQ16
    vbd17 = math.cos(x[81]) * vbD17 + math.sin(x[81]) * vbQ17
    vbq17 = -math.sin(x[81]) * vbD17 + math.cos(x[81]) * vbQ17
    vbd18 = math.cos(x[86]) * vbD18 + math.sin(x[86]) * vbQ18
    vbq18 = -math.sin(x[86]) * vbD18 + math.cos(x[86]) * vbQ18
    vbd19 = math.cos(x[91]) * vbD19 + math.sin(x[91]) * vbQ19
    vbq19 = -math.sin(x[91]) * vbD19 + math.cos(x[91]) * vbQ19
    vbd20 = math.cos(x[96]) * vbD20 + math.sin(x[96]) * vbQ20
    vbq20 = -math.sin(x[96]) * vbD20 + math.cos(x[96]) * vbQ20

    wcom = x[161] - mp1 * x[2]
    # ---------------DG1--------------------
    vod1_star = x[181] - nq1 * x[3]
    voq1_star = 0
    xdot1 = x[161] - mp1 * x[2] - wcom
    xdot2 = wc * (vod1_star * x[4] + voq1_star * x[5] - x[2])
    xdot3 = wc * (-vod1_star * x[5] + voq1_star * x[4] - x[3])
    xdot4 = (-rLc / Lc) * x[4] + wcom * x[5] + (1 / Lc) * (vod1_star - vbd1)
    xdot5 = (-rLc / Lc) * x[5] - wcom * x[4] + (1 / Lc) * (voq1_star - vbq1)

    # ----------------DG2-------------------
    vod2_star = x[182] - nq2 * x[8]
    voq2_star = 0
    xdot6 = x[162] - mp2 * x[7] - wcom
    xdot7 = wc * (vod2_star * x[9] + voq2_star * x[10] - x[7])
    xdot8 = wc * (-vod2_star * x[10] + voq2_star * x[9] - x[8])
    xdot9 = (-rLc / Lc) * x[9] + wcom * x[10] + (1 / Lc) * (vod2_star - vbd2)
    xdot10 = (-rLc / Lc) * x[10] - wcom * x[9] + (1 / Lc) * (voq2_star - vbq2)

    # ----------------DG3-------------------
    vod3_star = x[183] - nq3 * x[13]
    voq3_star = 0
    xdot11 = x[163] - mp3 * x[12] - wcom
    xdot12 = wc * (vod3_star * x[14] + voq3_star * x[15] - x[12])
    xdot13 = wc * (-vod3_star * x[15] + voq3_star * x[14] - x[13])
    xdot14 = (-rLc / Lc) * x[14] + wcom * x[15] + (1 / Lc) * (vod3_star - vbd3)
    xdot15 = (-rLc / Lc) * x[15] - wcom * x[14] + (1 / Lc) * (voq3_star - vbq3)

    # ----------------DG4-------------------
    vod4_star = x[184] - nq4 * x[18]
    voq4_star = 0
    xdot16 = x[164] - mp4 * x[17] - wcom
    xdot17 = wc * (vod4_star * x[19] + voq4_star * x[20] - x[17])
    xdot18 = wc * (-vod4_star * x[20] + voq4_star * x[19] - x[18])
    xdot19 = (-rLc / Lc) * x[19] + wcom * x[20] + (1 / Lc) * (vod4_star - vbd4)
    xdot20 = (-rLc / Lc) * x[20] - wcom * x[19] + (1 / Lc) * (voq4_star - vbq4)

    # ----------------DG5-------------------
    vod5_star = x[185] - nq5 * x[23]
    voq5_star = 0
    xdot21 = x[165] - mp5 * x[22] - wcom
    xdot22 = wc * (vod5_star * x[24] + voq5_star * x[25] - x[22])
    xdot23 = wc * (-vod5_star * x[25] + voq5_star * x[24] - x[23])
    xdot24 = (-rLc / Lc) * x[24] + wcom * x[25] + (1 / Lc) * (vod5_star - vbd5)
    xdot25 = (-rLc / Lc) * x[25] - wcom * x[24] + (1 / Lc) * (voq5_star - vbq5)

    # ----------------DG6-------------------
    vod6_star = x[186] - nq6 * x[28]
    voq6_star = 0
    xdot26 = x[166] - mp6 * x[27] - wcom
    xdot27 = wc * (vod6_star * x[29] + voq6_star * x[30] - x[27])
    xdot28 = wc * (-vod6_star * x[30] + voq6_star * x[29] - x[28])
    xdot29 = (-rLc / Lc) * x[29] + wcom * x[30] + (1 / Lc) * (vod6_star - vbd6)
    xdot30 = (-rLc / Lc) * x[30] - wcom * x[29] + (1 / Lc) * (voq6_star - vbq6)

    # ----------------DG7-------------------
    vod7_star = x[187] - nq7 * x[33]
    voq7_star = 0
    xdot31 = x[167] - mp7 * x[32] - wcom
    xdot32 = wc * (vod7_star * x[34] + voq7_star * x[35] - x[32])
    xdot33 = wc * (-vod7_star * x[35] + voq7_star * x[34] - x[33])
    xdot34 = (-rLc / Lc) * x[34] + wcom * x[35] + (1 / Lc) * (vod7_star - vbd7)
    xdot35 = (-rLc / Lc) * x[35] - wcom * x[34] + (1 / Lc) * (voq7_star - vbq7)

    # ----------------DG8-------------------
    vod8_star = x[188] - nq8 * x[38]
    voq8_star = 0
    xdot36 = x[168] - mp8 * x[37] - wcom
    xdot37 = wc * (vod8_star * x[39] + voq8_star * x[40] - x[37])
    xdot38 = wc * (-vod8_star * x[40] + voq8_star * x[39] - x[38])
    xdot39 = (-rLc / Lc) * x[39] + wcom * x[40] + (1 / Lc) * (vod8_star - vbd8)
    xdot40 = (-rLc / Lc) * x[40] - wcom * x[39] + (1 / Lc) * (voq8_star - vbq8)

    # ----------------DG9-------------------
    vod9_star = x[189] - nq9 * x[43]
    voq9_star = 0
    xdot41 = x[169] - mp9 * x[42] - wcom
    xdot42 = wc * (vod9_star * x[44] + voq9_star * x[45] - x[42])
    xdot43 = wc * (-vod9_star * x[45] + voq9_star * x[44] - x[43])
    xdot44 = (-rLc / Lc) * x[44] + wcom * x[45] + (1 / Lc) * (vod9_star - vbd9)
    xdot45 = (-rLc / Lc) * x[45] - wcom * x[44] + (1 / Lc) * (voq9_star - vbq9)

    # ----------------DG10-------------------
    vod10_star = x[190] - nq10 * x[48]
    voq10_star = 0
    xdot46 = x[170] - mp10 * x[47] - wcom
    xdot47 = wc * (vod10_star * x[49] + voq10_star * x[50] - x[47])
    xdot48 = wc * (-vod10_star * x[50] + voq10_star * x[49] - x[48])
    xdot49 = (-rLc / Lc) * x[49] + wcom * x[50] + (1 / Lc) * (vod10_star - vbd10)
    xdot50 = (-rLc / Lc) * x[50] - wcom * x[49] + (1 / Lc) * (voq10_star - vbq10)

    # ----------------DG11-------------------
    vod11_star = x[191] - nq11 * x[53]
    voq11_star = 0
    xdot51 = x[171] - mp11 * x[52] - wcom
    xdot52 = wc * (vod11_star * x[54] + voq11_star * x[55] - x[52])
    xdot53 = wc * (-vod11_star * x[55] + voq11_star * x[54] - x[53])
    xdot54 = (-rLc / Lc) * x[54] + wcom * x[55] + (1 / Lc) * (vod11_star - vbd11)
    xdot55 = (-rLc / Lc) * x[55] - wcom * x[54] + (1 / Lc) * (voq11_star - vbq11)

    # ----------------DG12-------------------
    vod12_star = x[192] - nq12 * x[58]
    voq12_star = 0
    xdot56 = x[172] - mp12 * x[57] - wcom
    xdot57 = wc * (vod12_star * x[59] + voq12_star * x[60] - x[57])
    xdot58 = wc * (-vod12_star * x[60] + voq12_star * x[59] - x[58])
    xdot59 = (-rLc / Lc) * x[59] + wcom * x[60] + (1 / Lc) * (vod12_star - vbd12)
    xdot60 = (-rLc / Lc) * x[60] - wcom * x[59] + (1 / Lc) * (voq12_star - vbq12)

    # ----------------DG13-------------------
    vod13_star = x[193] - nq13 * x[63]
    voq13_star = 0
    xdot61 = x[173] - mp13 * x[62] - wcom
    xdot62 = wc * (vod13_star * x[64] + voq13_star * x[65] - x[62])
    xdot63 = wc * (-vod13_star * x[65] + voq13_star * x[54] - x[63])
    xdot64 = (-rLc / Lc) * x[64] + wcom * x[65] + (1 / Lc) * (vod13_star - vbd13)
    xdot65 = (-rLc / Lc) * x[65] - wcom * x[64] + (1 / Lc) * (voq13_star - vbq13)

    # ----------------DG14-------------------
    vod14_star = x[194] - nq14 * x[68]
    voq14_star = 0
    xdot66 = x[174] - mp14 * x[67] - wcom
    xdot67 = wc * (vod14_star * x[69] + voq14_star * x[70] - x[67])
    xdot68 = wc * (-vod14_star * x[70] + voq14_star * x[69] - x[68])
    xdot69 = (-rLc / Lc) * x[69] + wcom * x[70] + (1 / Lc) * (vod14_star - vbd14)
    xdot70 = (-rLc / Lc) * x[70] - wcom * x[69] + (1 / Lc) * (voq14_star - vbq14)

    # ----------------DG15-------------------
    vod15_star = x[195] - nq15 * x[73]
    voq15_star = 0
    xdot71 = x[175] - mp15 * x[72] - wcom
    xdot72 = wc * (vod15_star * x[74] + voq15_star * x[75] - x[72])
    xdot73 = wc * (-vod15_star * x[75] + voq15_star * x[74] - x[73])
    xdot74 = (-rLc / Lc) * x[74] + wcom * x[75] + (1 / Lc) * (vod15_star - vbd15)
    xdot75 = (-rLc / Lc) * x[75] - wcom * x[74] + (1 / Lc) * (voq15_star - vbq15)

    # ----------------DG16-------------------
    vod16_star = x[196] - nq16 * x[78]
    voq16_star = 0
    xdot76 = x[176] - mp16 * x[77] - wcom
    xdot77 = wc * (vod16_star * x[79] + voq16_star * x[80] - x[77])
    xdot78 = wc * (-vod16_star * x[80] + voq16_star * x[79] - x[78])
    xdot79 = (-rLc / Lc) * x[79] + wcom * x[80] + (1 / Lc) * (vod16_star - vbd16)
    xdot80 = (-rLc / Lc) * x[80] - wcom * x[79] + (1 / Lc) * (voq16_star - vbq16)

    # ----------------DG17-------------------
    vod17_star = x[197] - nq17 * x[83]
    voq17_star = 0
    xdot81 = x[177] - mp17 * x[82] - wcom
    xdot82 = wc * (vod17_star * x[84] + voq17_star * x[85] - x[82])
    xdot83 = wc * (-vod17_star * x[85] + voq17_star * x[84] - x[83])
    xdot84 = (-rLc / Lc) * x[84] + wcom * x[85] + (1 / Lc) * (vod17_star - vbd17)
    xdot85 = (-rLc / Lc) * x[85] - wcom * x[84] + (1 / Lc) * (voq17_star - vbq17)

    # ----------------DG18-------------------
    vod18_star = x[198] - nq18 * x[88]
    voq18_star = 0
    xdot86 = x[178] - mp18 * x[87] - wcom
    xdot87 = wc * (vod18_star * x[89] + voq18_star * x[90] - x[87])
    xdot88 = wc * (-vod18_star * x[90] + voq18_star * x[89] - x[88])
    xdot89 = (-rLc / Lc) * x[89] + wcom * x[90] + (1 / Lc) * (vod18_star - vbd18)
    xdot90 = (-rLc / Lc) * x[90] - wcom * x[89] + (1 / Lc) * (voq18_star - vbq18)

    # ----------------DG19-------------------
    vod19_star = x[199] - nq19 * x[93]
    voq19_star = 0
    xdot91 = x[179] - mp19 * x[92] - wcom
    xdot92 = wc * (vod19_star * x[94] + voq19_star * x[95] - x[92])
    xdot93 = wc * (-vod19_star * x[95] + voq19_star * x[94] - x[93])
    xdot94 = (-rLc / Lc) * x[94] + wcom * x[95] + (1 / Lc) * (vod19_star - vbd19)
    xdot95 = (-rLc / Lc) * x[95] - wcom * x[94] + (1 / Lc) * (voq19_star - vbq19)

    # ----------------DG20-------------------
    vod20_star = x[200] - nq20 * x[98]
    voq20_star = 0
    xdot96 = x[180] - mp20 * x[97] - wcom
    xdot97 = wc * (vod20_star * x[99] + voq20_star * x[100] - x[97])
    xdot98 = wc * (-vod20_star * x[100] + voq20_star * x[99] - x[98])
    xdot99 = (-rLc / Lc) * x[99] + wcom * x[100] + (1 / Lc) * (vod20_star - vbd20)
    xdot100 = (-rLc / Lc) * x[100] - wcom * x[99] + (1 / Lc) * (voq20_star - vbq20)

    # -------------------------Lines------------------
    # ----1 -> 6-----
    xdot101 = (-rline1 / Lline1) * x[101] + wcom * x[102] + (1 / Lline1) * (vbD1 - vbD6)
    xdot102 = (-rline1 / Lline1) * x[102] - wcom * x[101] + (1 / Lline1) * (vbQ1 - vbQ6)
    # ----6 -> 7-----
    xdot103 = (-rline2 / Lline2) * x[103] + wcom * x[104] + (1 / Lline2) * (vbD6 - vbD7)
    xdot104 = (-rline2 / Lline2) * x[104] - wcom * x[103] + (1 / Lline2) * (vbQ6 - vbQ7)
    # ----7 -> 2-----
    xdot105 = (-rline3 / Lline3) * x[105] + wcom * x[106] + (1 / Lline3) * (vbD7 - vbD2)
    xdot106 = (-rline3 / Lline3) * x[106] - wcom * x[105] + (1 / Lline3) * (vbQ7 - vbQ2)
    # ----2 -> 3-----
    xdot107 = (-rline4 / Lline4) * x[107] + wcom * x[108] + (1 / Lline4) * (vbD2 - vbD3)
    xdot108 = (-rline4 / Lline4) * x[108] - wcom * x[107] + (1 / Lline4) * (vbQ2 - vbQ3)
    # ----3 -> 8-----
    xdot109 = (-rline5 / Lline5) * x[109] + wcom * x[110] + (1 / Lline5) * (vbD3 - vbD8)
    xdot110 = (-rline5 / Lline5) * x[110] - wcom * x[109] + (1 / Lline5) * (vbQ3 - vbQ8)
    # ----8 -> 9-----
    xdot111 = (-rline6 / Lline6) * x[111] + wcom * x[112] + (1 / Lline6) * (vbD8 - vbD9)
    xdot112 = (-rline6 / Lline6) * x[112] - wcom * x[111] + (1 / Lline6) * (vbQ8 - vbQ9)
    # ----9 -> 4-----
    xdot113 = (-rline7 / Lline7) * x[113] + wcom * x[114] + (1 / Lline7) * (vbD9 - vbD4)
    xdot114 = (-rline7 / Lline7) * x[114] - wcom * x[113] + (1 / Lline7) * (vbQ9 - vbQ4)
    # ----4 -> 5-----
    xdot115 = (-rline8 / Lline8) * x[115] + wcom * x[116] + (1 / Lline8) * (vbD4 - vbD5)
    xdot116 = (-rline8 / Lline8) * x[116] - wcom * x[115] + (1 / Lline8) * (vbQ4 - vbQ5)
    # ----5 -> 10-----
    xdot117 = (-rline9 / Lline9) * x[117] + wcom * x[118] + (1 / Lline9) * (vbD5 - vbD10)
    xdot118 = (-rline9 / Lline9) * x[118] - wcom * x[117] + (1 / Lline9) * (vbQ5 - vbQ10)
    # ----8 -> 11-----
    xdot119 = (-rline10 / Lline10) * x[119] + wcom * x[120] + (1 / Lline10) * (vbD8 - vbD11)
    xdot120 = (-rline10 / Lline10) * x[120] - wcom * x[119] + (1 / Lline10) * (vbQ8 - vbQ11)
    # ----10 -> 13-----
    xdot121 = (-rline11 / Lline11) * x[121] + wcom * x[122] + (1 / Lline11) * (vbD10 - vbD13)
    xdot122 = (-rline11 / Lline11) * x[122] - wcom * x[121] + (1 / Lline11) * (vbQ10 - vbQ13)
    # ----11 -> 16-----
    xdot123 = (-rline12 / Lline12) * x[123] + wcom * x[124] + (1 / Lline12) * (vbD11 - vbD16)
    xdot124 = (-rline12 / Lline12) * x[124] - wcom * x[123] + (1 / Lline12) * (vbQ11 - vbQ16)
    # ----16 -> 17-----
    xdot125 = (-rline13 / Lline13) * x[125] + wcom * x[126] + (1 / Lline13) * (vbD16 - vbD17)
    xdot126 = (-rline13 / Lline13) * x[126] - wcom * x[125] + (1 / Lline13) * (vbQ16 - vbQ17)
    # ----17 -> 12-----
    xdot127 = (-rline14 / Lline14) * x[127] + wcom * x[128] + (1 / Lline14) * (vbD17 - vbD12)
    xdot128 = (-rline14 / Lline14) * x[128] - wcom * x[127] + (1 / Lline14) * (vbQ17 - vbQ12)
    # ----12 -> 13-----
    xdot129 = (-rline15 / Lline15) * x[129] + wcom * x[130] + (1 / Lline15) * (vbD12 - vbD13)
    xdot130 = (-rline15 / Lline15) * x[130] - wcom * x[129] + (1 / Lline15) * (vbQ12 - vbQ13)
    # ----13 -> 18-----
    xdot131 = (-rline16 / Lline16) * x[131] + wcom * x[132] + (1 / Lline16) * (vbD13 - vbD18)
    xdot132 = (-rline16 / Lline16) * x[132] - wcom * x[131] + (1 / Lline16) * (vbQ13 - vbQ18)
    # ----18 -> 19-----
    xdot133 = (-rline17 / Lline17) * x[133] + wcom * x[134] + (1 / Lline17) * (vbD18 - vbD19)
    xdot134 = (-rline17 / Lline17) * x[134] - wcom * x[133] + (1 / Lline17) * (vbQ18 - vbQ19)
    # ----19 -> 14-----
    xdot135 = (-rline18 / Lline18) * x[135] + wcom * x[136] + (1 / Lline18) * (vbD19 - vbD14)
    xdot136 = (-rline18 / Lline18) * x[136] - wcom * x[135] + (1 / Lline18) * (vbQ19 - vbQ14)
    # ----14 -> 15-----
    xdot137 = (-rline19 / Lline19) * x[137] + wcom * x[138] + (1 / Lline19) * (vbD14 - vbD15)
    xdot138 = (-rline19 / Lline19) * x[138] - wcom * x[137] + (1 / Lline19) * (vbQ14 - vbQ15)
    # ----15 -> 20-----
    xdot139 = (-rline20 / Lline20) * x[139] + wcom * x[140] + (1 / Lline20) * (vbD15 - vbD20)
    xdot140 = (-rline20 / Lline20) * x[140] - wcom * x[139] + (1 / Lline20) * (vbQ15 - vbQ20)

    # -------------------------Loads------------------
    # ------Load1--------
    xdot141 = (-disturbance_R[0] * Rload1 / (disturbance_L[0] * Lload1)) * x[141] + wcom * x[142] + (
                1 / (disturbance_L[0] * Lload1)) * vbD1
    xdot142 = (-disturbance_R[0] * Rload1 / (disturbance_L[0] * Lload1)) * x[142] - wcom * x[141] + (
                1 / (disturbance_L[0] * Lload1)) * vbQ1
    # ------Load2--------
    xdot143 = (-disturbance_R[1] * Rload2 / (disturbance_L[1] * Lload2)) * x[143] + wcom * x[144] + (
                1 / (disturbance_L[1] * Lload2)) * vbD7
    xdot144 = (-disturbance_R[1] * Rload2 / (disturbance_L[1] * Lload2)) * x[144] - wcom * x[143] + (
                1 / (disturbance_L[1] * Lload2)) * vbQ7
    # ------Load3--------
    xdot145 = (-disturbance_R[2] * Rload3 / (disturbance_L[2] * Lload3)) * x[145] + wcom * x[146] + (
                1 / (disturbance_L[2] * Lload3)) * vbD3
    xdot146 = (-disturbance_R[2] * Rload3 / (disturbance_L[2] * Lload3)) * x[146] - wcom * x[145] + (
                1 / (disturbance_L[2] * Lload3)) * vbQ3
    # ------Load4--------
    xdot147 = (-disturbance_R[3] * Rload4 / (disturbance_L[3] * Lload4)) * x[147] + wcom * x[148] + (
                1 / (disturbance_L[3] * Lload4)) * vbD9
    xdot148 = (-disturbance_R[3] * Rload4 / (disturbance_L[3] * Lload4)) * x[148] - wcom * x[147] + (
                1 / (disturbance_L[3] * Lload4)) * vbQ9
    # ------Load5--------
    xdot149 = (-disturbance_R[4] * Rload5 / (disturbance_L[4] * Lload5)) * x[149] + wcom * x[150] + (
                1 / (disturbance_L[4] * Lload5)) * vbD5
    xdot150 = (-disturbance_R[4] * Rload5 / (disturbance_L[4] * Lload5)) * x[150] - wcom * x[149] + (
                1 / (disturbance_L[4] * Lload5)) * vbQ5
    # ------Load6--------
    xdot151 = (-disturbance_R[5] * Rload6 / (disturbance_L[5] * Lload6)) * x[151] + wcom * x[152] + (
                1 / (disturbance_L[5] * Lload6)) * vbD11
    xdot152 = (-disturbance_R[5] * Rload6 / (disturbance_L[5] * Lload6)) * x[152] - wcom * x[151] + (
                1 / (disturbance_L[5] * Lload6)) * vbQ11
    # ------Load7--------
    xdot153 = (-disturbance_R[6] * Rload7 / (disturbance_L[6] * Lload7)) * x[153] + wcom * x[154] + (
                1 / (disturbance_L[6] * Lload7)) * vbD17
    xdot154 = (-disturbance_R[6] * Rload7 / (disturbance_L[6] * Lload7)) * x[154] - wcom * x[153] + (
                1 / (disturbance_L[6] * Lload7)) * vbQ17
    # ------Load8--------
    xdot155 = (-disturbance_R[7] * Rload8 / (disturbance_L[7] * Lload8)) * x[155] + wcom * x[156] + (
                1 / (disturbance_L[7] * Lload8)) * vbD13
    xdot156 = (-disturbance_R[7] * Rload8 / (disturbance_L[7] * Lload8)) * x[156] - wcom * x[155] + (
                1 / (disturbance_L[7] * Lload8)) * vbQ13
    # ------Load9--------
    xdot157 = (-disturbance_R[8] * Rload9 / (disturbance_L[8] * Lload9)) * x[157] + wcom * x[158] + (
                1 / (disturbance_L[8] * Lload9)) * vbD19
    xdot158 = (-disturbance_R[8] * Rload9 / (disturbance_L[8] * Lload9)) * x[158] - wcom * x[157] + (
                1 / (disturbance_L[8] * Lload9)) * vbQ19
    # ------Load10--------
    xdot159 = (-disturbance_R[9] * Rload10 / (disturbance_L[9] * Lload10)) * x[159] + wcom * x[160] + (
                1 / (disturbance_L[9] * Lload10)) * vbD15
    xdot160 = (-disturbance_R[9] * Rload10 / (disturbance_L[9] * Lload10)) * x[160] - wcom * x[159] + (
                1 / (disturbance_L[9] * Lload10)) * vbQ15

    # ----------------------------------------------------
    # Controller Parameters
    if t <= 0.4:
        # this time is for the primary control
        xdot161 = 0
        xdot162 = 0
        xdot163 = 0
        xdot164 = 0
        xdot165 = 0
        xdot166 = 0
        xdot167 = 0
        xdot168 = 0
        xdot169 = 0
        xdot170 = 0
        xdot171 = 0
        xdot172 = 0
        xdot173 = 0
        xdot174 = 0
        xdot175 = 0
        xdot176 = 0
        xdot177 = 0
        xdot178 = 0
        xdot179 = 0
        xdot180 = 0
        xdot181 = 0
        xdot182 = 0
        xdot183 = 0
        xdot184 = 0
        xdot185 = 0
        xdot186 = 0
        xdot187 = 0
        xdot188 = 0
        xdot189 = 0
        xdot190 = 0
        xdot191 = 0
        xdot192 = 0
        xdot193 = 0
        xdot194 = 0
        xdot195 = 0
        xdot196 = 0
        xdot197 = 0
        xdot198 = 0
        xdot199 = 0
        xdot200 = 0
    else:
        Pratio = np.array([[mp1 * x[2]], [mp2 * x[7]], [mp3 * x[12]], [mp4 * x[17]], [mp5 * x[22]],
                           [mp6 * x[27]], [mp7 * x[37]], [mp8 * x[42]], [mp9 * x[47]], [mp10 * x[52]],
                           [mp11 * x[57]], [mp12 * x[62]], [mp13 * x[67]], [mp14 * x[72]], [mp15 * x[77]],
                           [mp16 * x[82]], [mp17 * x[87]], [mp18 * x[92]], [mp19 * x[97]], [mp20 * x[102]]])

        w_array = np.array(
            [[x[161] - Pratio[0][0]], [x[162] - Pratio[1][0]], [x[163] - Pratio[2][0]], [x[164] - Pratio[3][0]],
             [x[165] - Pratio[4][0]], [x[166] - Pratio[5][0]], [x[167] - Pratio[6][0]], [x[168] - Pratio[7][0]],
             [x[169] - Pratio[8][0]], [x[170] - Pratio[9][0]], [x[171] - Pratio[10][0]], [x[172] - Pratio[11][0]],
             [x[173] - Pratio[12][0]], [x[174] - Pratio[13][0]], [x[175] - Pratio[14][0]], [x[176] - Pratio[15][0]],
             [x[177] - Pratio[16][0]], [x[178] - Pratio[17][0]], [x[179] - Pratio[18][0]], [x[180] - Pratio[19][0]]])

        # Conventional Freq Control
        Synch_Mat = -1 * a_ctrl * (
                np.dot(L + G, w_array - np.array([[wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref],
                                                  [wref], [wref], [wref], [wref]])) + np.dot(L, Pratio))

        # ---frequency input---
        xdot161 = Synch_Mat[0][0]
        xdot162 = Synch_Mat[1][0]
        xdot163 = Synch_Mat[2][0]
        xdot164 = Synch_Mat[3][0]
        xdot165 = Synch_Mat[4][0]
        xdot166 = Synch_Mat[5][0]
        xdot167 = Synch_Mat[6][0]
        xdot168 = Synch_Mat[7][0]
        xdot169 = Synch_Mat[8][0]
        xdot170 = Synch_Mat[9][0]
        xdot171 = Synch_Mat[10][0]
        xdot172 = Synch_Mat[11][0]
        xdot173 = Synch_Mat[12][0]
        xdot174 = Synch_Mat[13][0]
        xdot175 = Synch_Mat[14][0]
        xdot176 = Synch_Mat[15][0]
        xdot177 = Synch_Mat[16][0]
        xdot178 = Synch_Mat[17][0]
        xdot179 = Synch_Mat[18][0]
        xdot180 = Synch_Mat[19][0]
        # ---voltage input---
        xdot181 = 0
        xdot182 = 0
        xdot183 = 0
        xdot184 = 0
        xdot185 = 0
        xdot186 = 0
        xdot187 = 0
        xdot188 = 0
        xdot189 = 0
        xdot190 = 0
        xdot191 = 0
        xdot192 = 0
        xdot193 = 0
        xdot194 = 0
        xdot195 = 0
        xdot196 = 0
        xdot197 = 0
        xdot198 = 0
        xdot199 = 0
        xdot200 = 0

    return np.array(
        [0, xdot1, xdot2, xdot3, xdot4, xdot5, xdot6, xdot7, xdot8, xdot9, xdot10, xdot11, xdot12, xdot13, xdot14,
         xdot15, xdot16, xdot17, xdot18, xdot19, xdot20, xdot21, xdot22, xdot23, xdot24, xdot25, xdot26, xdot27,
         xdot28, xdot29, xdot30, xdot31, xdot32, xdot33, xdot34, xdot35, xdot36, xdot37, xdot38, xdot39, xdot40,
         xdot41, xdot42, xdot43, xdot44, xdot45, xdot46, xdot47, xdot48, xdot49, xdot50, xdot51, xdot52, xdot53,
         xdot54, xdot55, xdot56, xdot57, xdot58, xdot59, xdot60, xdot61, xdot62, xdot63, xdot64, xdot65, xdot66,
         xdot67, xdot68, xdot69, xdot70, xdot71, xdot72, xdot73, xdot74, xdot75, xdot76, xdot77, xdot78, xdot79,
         xdot80, xdot81, xdot82, xdot83, xdot84, xdot85, xdot86, xdot87, xdot88, xdot89, xdot90, xdot91, xdot92,
         xdot93, xdot94, xdot95, xdot96, xdot97, xdot98, xdot99, xdot100, xdot101, xdot102, xdot103, xdot104,
         xdot105, xdot106, xdot107, xdot108, xdot109, xdot110, xdot111, xdot112, xdot113, xdot114, xdot115,
         xdot116, xdot117, xdot118, xdot119, xdot120, xdot121, xdot122, xdot123, xdot124, xdot125, xdot126,
         xdot127, xdot128, xdot129, xdot130, xdot131, xdot132, xdot133, xdot134, xdot135, xdot136, xdot137,
         xdot138, xdot139, xdot140, xdot141, xdot142, xdot143, xdot144, xdot145, xdot146, xdot147, xdot148,
         xdot149, xdot150, xdot151, xdot152, xdot153, xdot154, xdot155, xdot156, xdot157, xdot158, xdot159,
         xdot160, xdot161, xdot162, xdot163, xdot164, xdot165, xdot166, xdot167, xdot168, xdot169, xdot170,
         xdot171, xdot172, xdot173, xdot174, xdot175, xdot176, xdot177, xdot178, xdot179, xdot180, xdot181,
         xdot182, xdot183, xdot184, xdot185, xdot186, xdot187, xdot188, xdot189, xdot190, xdot191, xdot192,
         xdot193, xdot194, xdot195, xdot196, xdot197, xdot198, xdot199, xdot200, ioD1, ioQ1, vbD1, vbQ1,
         ioD2, ioQ2, vbD2, vbQ2, ioD3, ioQ3, vbD3, vbQ3, ioD4, ioQ4, vbD4, vbQ4, ioD5, ioQ5, vbD5, vbQ5,
         ioD6, ioQ6, vbD6, vbQ6, ioD7, ioQ7, vbD7, vbQ7, ioD8, ioQ8, vbD8, vbQ8, ioD9, ioQ9, vbD9, vbQ9,
         ioD10, ioQ10, vbD10, vbQ10, ioD11, ioQ11, vbD11, vbQ11, ioD12, ioQ12, vbD12, vbQ13, ioD13, ioQ13, vbD13, vbQ13,
         ioD14, ioQ14, vbD14, vbQ14, ioD15, ioQ15, vbD15, vbQ15, ioD16, ioQ16, vbD16, vbQ16, ioD17, ioQ17, vbD17, vbQ17,
         ioD18, ioQ18, vbD18, vbQ18, ioD19, ioQ19, vbD19, vbQ19, ioD20, ioQ20, vbD20, vbQ20
         ])